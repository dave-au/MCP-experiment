#
# Simple MCP server for shell commands
#
# - Uses stdio
#
from mcp.server.fastmcp import FastMCP
import sys
import logging
import subprocess
import shlex
from pathlib import Path
from typing import List, Optional, Dict, Any
import os

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
#logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP("Shell")

# ---- Configuration ----
# Only allow these commands (keys are exposed names, values are actual executables looked up on PATH)
ALLOWLIST = {
    "ls": "ls",
    "pwd": "pwd",
    "cat": "cat",
    "echo": "echo",
}

# Base directory: only allow file paths inside here 
BASE_DIR = Path.cwd().resolve()

# Output/Runtime limits
MAX_OUTPUT_BYTES = 100_000      # cap stdout/stderr to 100 KB each
DEFAULT_TIMEOUT_SEC = 10        # kill long-running processes


def _is_path_safe(p: Optional[str]) -> bool:
    if p is None or p == "":
        return True
    try:
        return Path(p).resolve().is_relative_to(BASE_DIR)
    except AttributeError:
        # Python < 3.9 fallback
        rp = Path(p).resolve()
        return str(rp).startswith(str(BASE_DIR))


def _truncate(b: bytes) -> bytes:
    return b if len(b) <= MAX_OUTPUT_BYTES else b[:MAX_OUTPUT_BYTES]


def _which(exe: str) -> Optional[str]:
    # Minimal which to avoid external deps
    paths = os.environ.get("PATH", "").split(os.pathsep)
    exts = [""] if os.name != "nt" else os.environ.get("PATHEXT", ".EXE;.BAT;.CMD").split(";")
    for d in paths:
        candidate = Path(d, exe)
        if os.name == "nt":
            for ext in exts:
                c2 = candidate.with_suffix(ext.lower())
                if c2.exists() and c2.is_file():
                    return str(c2)
        else:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


@mcp.tool(name="list_allowed", description="List allowed commands and the sandbox base directory.")
def list_allowed() -> Dict[str, Any]:
    """Return info about the sandbox and which commands are permitted."""
    return {
        "base_dir": str(BASE_DIR),
        "allowed": sorted(ALLOWLIST.keys()),
        "timeout_sec": DEFAULT_TIMEOUT_SEC,
        "max_output_bytes": MAX_OUTPUT_BYTES,
    }


@mcp.tool(
    name="exec",
    description=(
        "Execute a sandboxed OS command from an allow-list. "
        "Arguments are passed as an array; no shell expansion is performed."
    ),
)
def exec_cmd(
    command: str,
    args: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    timeout_sec: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run a permitted command safely.

    Parameters:
    - command: one of the allow-listed command names (e.g., 'ls', 'cat').
    - args: list of arguments (each item is a single token; do not pass a single string like 'a b').
    - cwd: working directory; must be inside BASE_DIR (defaults to BASE_DIR).
    - timeout_sec: optional override (defaults to DEFAULT_TIMEOUT_SEC).
    - env: optional extra environment variables (merged with a minimal safe env).
    """
    # Validate command
    if command not in ALLOWLIST:
        return {
            "ok": False,
            "error": f"Command '{command}' is not allowed. Use list_allowed to see permitted commands."
        }

    resolved_exe = _which(ALLOWLIST[command])
    if resolved_exe is None:
        return {"ok": False, "error": f"Executable for '{command}' not found on PATH."}

    # Validate and set cwd
    safe_cwd = str(BASE_DIR)
    if cwd:
        # Convert relative paths to absolute relative to BASE_DIR
        candidate = (BASE_DIR / cwd).resolve() if not Path(cwd).is_absolute() else Path(cwd).resolve()
        if not _is_path_safe(str(candidate)):
            return {"ok": False, "error": "cwd is outside sandbox base directory."}
        safe_cwd = str(candidate)

    # Build argv (no shell=True!)
    argv = [resolved_exe] + (args or [])

    # Minimal, sanitized environment
    safe_env = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    if env:
        # allow adding specific safe env vars; avoid overwriting PATH unless you intend to
        for k, v in env.items():
            if k.upper() not in {"LD_PRELOAD", "LD_LIBRARY_PATH"}:
                safe_env[k] = v

    # Timeout
    tmo = timeout_sec if (isinstance(timeout_sec, int) and timeout_sec > 0) else DEFAULT_TIMEOUT_SEC

    try:
        proc = subprocess.run(
            argv,
            cwd=safe_cwd,
            env=safe_env,
            capture_output=True,
            text=False,            # capture as bytes, then truncate cleanly
            timeout=tmo,
            shell=False,           # critical: no shell interpolation
        )
        out = _truncate(proc.stdout or b"").decode(errors="replace")
        err = _truncate(proc.stderr or b"").decode(errors="replace")
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": out,
            "stderr": err,
            "cwd": safe_cwd,
            "argv": argv,
        }
    except subprocess.TimeoutExpired as e:
        out = _truncate(e.stdout or b"").decode(errors="replace") if hasattr(e, "stdout") else ""
        err = _truncate(e.stderr or b"").decode(errors="replace") if hasattr(e, "stderr") else ""
        return {
            "ok": False,
            "error": f"Timeout after {tmo}s",
            "stdout": out,
            "stderr": err,
            "cwd": safe_cwd,
            "argv": argv,
        }
    except Exception as e:
        return {"ok": False, "error": repr(e), "cwd": safe_cwd, "argv": argv}


if __name__ == "__main__":
    mcp.run(transport="stdio")
