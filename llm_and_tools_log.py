# llm_and_tools_log.py
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import json

# Back/forward compatible imports
try:
    from langchain_core.callbacks import BaseCallbackHandler  # LC >= 0.2
    from langchain_core.messages import BaseMessage
except Exception:  # pragma: no cover
    from langchain.callbacks.base import BaseCallbackHandler  # older LC
    try:
        from langchain.schema import BaseMessage  # older LC
    except Exception:  # pragma: no cover
        BaseMessage = object  # fallback

# ---------- small helpers ----------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def _redact(text: str, secrets: List[str]) -> str:
    for token in secrets:
        if token:
            text = text.replace(token, "****REDACTED****")
    return text

def _maybe_pretty(obj: Any, pretty: bool) -> str:
    """JSON-dump if possible; pretty-print when requested, else compact."""
    try:
        if pretty:
            return json.dumps(obj, ensure_ascii=False, indent=2)
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(obj)

def _safe_to_text(obj: Any, pretty: bool) -> str:
    """Best-effort conversion of tool args/results/messages to text."""
    # LangChain messages (e.g., ToolMessage)
    if isinstance(obj, BaseMessage):
        return str(getattr(obj, "content", ""))

    # Bytes → decode
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")

    # Dict/List → JSON
    if isinstance(obj, (dict, list)):
        return _maybe_pretty(obj, pretty)

    # Pydantic-style
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return _maybe_pretty(dump(), pretty)
        except Exception:
            pass

    # String already
    if isinstance(obj, str):
        return obj

    # Fallback
    try:
        return str(obj)
    except Exception:
        return f"<unprintable {type(obj).__name__}>"

def _truncate_text(s: str, limit: Optional[int]) -> str:
    if limit is None or not isinstance(s, str) or len(s) <= limit:
        return s
    return s[:limit] + f"\n... [truncated {len(s) - limit} chars]"

# ---------- main logger ----------

class LLMAndToolLogger(BaseCallbackHandler):
    """
    Logs:
      - LLM prompts (messages) and responses
      - Tool specs provided to the LLM (names + JSON schema, truncated)
      - Tool choice / tool_calls / function_call in the LLM response
      - Tool start/end (name, args, result), with safe pretty-print & truncation
    """

    def __init__(
        self,
        logfile: str = "llm_and_tools_wire.log",
        also_print: bool = False,
        redact: Optional[List[str]] = None,
        pretty_json: bool = True,
        max_schema_chars: int = 4000,
        max_tool_result_chars: int = 4000,
    ):
        self.logfile = logfile
        self.also_print = also_print
        self.redact = redact or []
        self.pretty = pretty_json
        self.max_schema_chars = max_schema_chars
        self.max_tool_result_chars = max_tool_result_chars

    # ---- I/O helper ----
    def _w(self, line: str):
        line = _redact(line, self.redact)
        stamped = f"[{_ts()}] {line}"
        with open(self.logfile, "a", encoding="utf-8") as f:
            f.write(stamped + "\n")
        if self.also_print:
            print(stamped)

    # ---- LLM (chat) hooks ----
    def on_chat_model_start(self, serialized: dict, messages, **kwargs: Any) -> None:
        # Messages sent to the model
        for i, convo in enumerate(messages):
            self._w(f"LLM REQUEST #{i+1} BEGIN")
            for msg in convo:
                role = getattr(msg, "type", getattr(msg, "role", "message")).upper()
                content = getattr(msg, "content", "")
                self._w(f"{role}: {content}")
            self._w(f"LLM REQUEST #{i+1} END")

        # Tool specs / tool_choice (when provided by the caller)
        inv = kwargs.get("invocation_params") or {}
        tools = inv.get("tools")
        tool_choice = inv.get("tool_choice") or inv.get("function_call")
        if tools:
            names: List[str] = []
            for t in tools:
                name = None
                if isinstance(t, dict):
                    fn = t.get("function")
                    if isinstance(fn, dict):
                        name = fn.get("name")
                    if not name:
                        name = t.get("name")
                names.append(name or "<unknown>")
            self._w(f"LLM TOOL-SPECS: {len(tools)} tools -> {names}")
            for t in tools:
                fn = t.get("function") if isinstance(t, dict) else None
                name = (fn or {}).get("name") if fn else (t.get("name") if isinstance(t, dict) else None)
                schema = (fn or {}).get("parameters") if fn else t
                schema_str = _maybe_pretty(schema, self.pretty)
                schema_str = _truncate_text(schema_str, self.max_schema_chars)
                self._w(f"LLM TOOL-SCHEMA {name or '<unknown>'}:\n{schema_str}")
        if tool_choice:
            self._w(f"LLM TOOL-CHOICE: {_maybe_pretty(tool_choice, self.pretty)}")

    def on_chat_model_end(self, response, **kwargs: Any) -> None:
        try:
            gens = getattr(response, "generations", []) or []
            for i, gen_list in enumerate(gens):
                if not gen_list:
                    continue
                msg = gen_list[0].message
                role = getattr(msg, "type", getattr(msg, "role", "AI")).upper()
                content = getattr(msg, "content", "")
                self._w(f"LLM RESPONSE #{i+1} BEGIN")
                self._w(f"{role}: {content}")

                ak = getattr(msg, "additional_kwargs", {}) or {}
                if "tool_calls" in ak:
                    self._w("LLM RESPONSE TOOL_CALLS:")
                    self._w(_maybe_pretty(ak["tool_calls"], self.pretty))
                if "function_call" in ak:  # legacy
                    self._w("LLM RESPONSE FUNCTION_CALL:")
                    self._w(_maybe_pretty(ak["function_call"], self.pretty))

                self._w(f"LLM RESPONSE #{i+1} END")
        except Exception as e:
            self._w(f"LLM RESPONSE (parse error): {e!r}")

        meta = getattr(response, "llm_output", {}) or {}
        usage = meta.get("token_usage") or meta.get("usage")
        if usage:
            self._w(f"USAGE: {usage}")

    # ---- Tool hooks ----
    def on_tool_start(self, serialized: Dict[str, Any], input_str: Union[str, Any], **kwargs: Any) -> None:
        name = (serialized.get("name")
                or serialized.get("kwargs", {}).get("name")
                or serialized.get("id")
                or kwargs.get("name")
                or "<tool>")
        # input_str may already be dict/obj or a JSON string
        value: Any = input_str
        if isinstance(input_str, str):
            try:
                value = json.loads(input_str)
            except Exception:
                value = input_str
        text = _safe_to_text(value, self.pretty)
        self._w(f"TOOL START {name} ARGS:\n{text}")

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        text = _safe_to_text(output, self.pretty)
        text = _truncate_text(text, self.max_tool_result_chars)
        self._w(f"TOOL END RESULT:\n{text}")

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        self._w(f"TOOL ERROR: {repr(error)}")
