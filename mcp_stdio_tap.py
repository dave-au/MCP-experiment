# mcp_stdio_tap.py  (file-only logging + optional pretty JSON)
import sys, os, asyncio, argparse, datetime, contextlib, json
from asyncio.subprocess import PIPE

def ts(): return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

def split_cmd(argv):
    if "--" in argv:
        i = argv.index("--"); return argv[:i], argv[i+1:]
    return [], argv

def make_logger(logfile: str|None, quiet: bool):
    fh = open(logfile, "a", encoding="utf-8", errors="replace") if logfile else None
    def write(s: str):
        if fh:
            fh.write(s); fh.flush()
        elif not quiet:
            # fall back to stderr only if no file and not quiet
            sys.__stderr__.write(s); sys.__stderr__.flush()
    def log(line: str):
        write(f"[{ts()}] {line}")
    def close():
        with contextlib.suppress(Exception):
            if fh: fh.close()
    return log, close

def prettify_if_json(line: str, enable: bool) -> str:
    """Try to pretty-print a single-line JSON-RPC frame; otherwise return original."""
    if not enable:
        return line
    s = line.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return line
    try:
        obj = json.loads(s)
        # Optional: only pretty-print JSON-RPC-shaped payloads
        if isinstance(obj, dict) and "jsonrpc" in obj:
            pretty = json.dumps(obj, indent=2, ensure_ascii=False)
            return pretty + "\n"
        return line
    except Exception:
        return line

async def pump_stdin_to_child(child_stdin, log, pretty: bool):
    """Client stdin -> Server stdin. Log as C -> S."""
    loop = asyncio.get_running_loop()
    r = asyncio.StreamReader(); p = asyncio.StreamReaderProtocol(r)
    await loop.connect_read_pipe(lambda: p, sys.stdin)
    pending = ""
    try:
        while True:
            chunk = await r.read(4096)
            if not chunk: break
            # Log decoded copy, line-aware with JSON pretty
            text = chunk.decode("utf-8", errors="replace")
            pending += text
            *lines, tail = pending.split("\n")
            for ln in lines:
                logged = prettify_if_json(ln + "\n", pretty)
                log(f"C -> S: {logged}")
            pending = tail
            # Forward raw bytes unchanged
            child_stdin.write(chunk); await child_stdin.drain()
    finally:
        with contextlib.suppress(Exception):
            child_stdin.close()

async def pump_child_to_stdout(child_stdout, log, pretty: bool):
    """Server stdout -> Client stdout. Log as S -> C."""
    pending = ""
    while True:
        chunk = await child_stdout.read(4096)
        if not chunk: break
        text = chunk.decode("utf-8", errors="replace")
        pending += text
        *lines, tail = pending.split("\n")
        for ln in lines:
            logged = prettify_if_json(ln + "\n", pretty)
            log(f"S -> C: {logged}")
        pending = tail
        # Forward to stdout unchanged
        sys.stdout.buffer.write(chunk); sys.stdout.flush()
    # Flush any leftover partial line into log (best effort)
    if pending:
        logged = prettify_if_json(pending, pretty)
        log(f"S -> C: {logged}")

async def pump_child_stderr(child_stderr, log, mirror: bool):
    """Server stderr -> Client stderr. Not JSON; just copy & log."""
    while True:
        chunk = await child_stderr.read(4096)
        if not chunk: break
        try:
            log(f"S-STDERR: {chunk.decode('utf-8', errors='replace')}")
        except Exception:
            log(f"S-STDERR: <binary {len(chunk)} bytes>\n")
        if mirror:
            sys.stderr.buffer.write(chunk); sys.stderr.flush()

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logfile", default=None, help="Write tap logs to this file")
    ap.add_argument("--quiet", action="store_true", help="Do not write tap logs to stderr")
    ap.add_argument("--no-mirror-child-stderr", action="store_true",
                    help="Do not forward server stderr to parent stderr (still logged)")
    ap.add_argument("--pretty", action="store_true",
                    help="Pretty-print line-delimited JSON frames in logs")
    args, rest = ap.parse_known_args()
    _, cmd = split_cmd(rest)
    if not cmd:
        print("Usage: mcp_stdio_tap.py [--logfile FILE] [--quiet] [--no-mirror-child-stderr] [--pretty] -- <cmd...>", file=sys.stderr)
        sys.exit(2)

    log, log_close = make_logger(args.logfile, args.quiet)
    log(f"launching: {' '.join(cmd)}\n")

    proc = await asyncio.create_subprocess_exec(*cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    pumps = [
        asyncio.create_task(pump_stdin_to_child(proc.stdin, log, args.pretty)),
        asyncio.create_task(pump_child_to_stdout(proc.stdout, log, args.pretty)),
        asyncio.create_task(pump_child_stderr(proc.stderr, log, mirror=not args.no_mirror_child_stderr)),
    ]
    rc = await proc.wait()
    for t in pumps:
        with contextlib.suppress(Exception): t.cancel()
    log(f"child exit: {rc}\n"); log_close(); sys.exit(rc)

if __name__ == "__main__":
    asyncio.run(main())

