# llm_log.py
from datetime import datetime
from typing import List, Any, Optional

# Back/forward compatible import for the callback base class
try:
    from langchain_core.callbacks import BaseCallbackHandler  # LangChain >= 0.2
except Exception:  # pragma: no cover
    from langchain.callbacks.base import BaseCallbackHandler  # older LangChain

class LLMTranscriptLogger(BaseCallbackHandler):
    """
    Logs chat prompts and responses to a file (no HTTP headers).
    Works with Chat models (on_chat_model_start/end) and falls back to LLM text hooks.
    """

    def __init__(self, logfile: str = "llm_wire.log", also_print: bool = False, redact: Optional[List[str]] = None):
        self.logfile = logfile
        self.also_print = also_print
        self.redact = redact or []

    # --- helpers ---
    def _w(self, text: str):
        # redact any provided tokens/strings
        for token in self.redact:
            if token:
                text = text.replace(token, "****REDACTED****")
        line = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {text}"
        with open(self.logfile, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.also_print:
            print(line)

    def _log_messages(self, prefix: str, messages):
        # messages: List[List[BaseMessage]] for chat models
        for i, convo in enumerate(messages):
            self._w(f"{prefix} #{i+1} BEGIN")
            for msg in convo:
                role = getattr(msg, "type", getattr(msg, "role", "message")).upper()
                content = getattr(msg, "content", "")
                self._w(f"{role}: {content}")
            self._w(f"{prefix} #{i+1} END")

    # --- preferred chat callbacks ---
    def on_chat_model_start(self, serialized: dict, messages, **kwargs: Any) -> None:
        self._log_messages("LLM REQUEST", messages)

    def on_chat_model_end(self, response, **kwargs: Any) -> None:
        # response.generations: List[List[ChatGeneration]]
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
                self._w(f"LLM RESPONSE #{i+1} END")
        except Exception as e:
            self._w(f"LLM RESPONSE (parse error): {e!r}")

        meta = getattr(response, "llm_output", {}) or {}
        usage = meta.get("token_usage") or meta.get("usage")
        if usage:
            self._w(f"USAGE: {usage}")

    # --- fallback for text-only models ---
    def on_llm_start(self, serialized: dict, prompts: List[str], **kwargs: Any) -> None:
        for i, p in enumerate(prompts):
            self._w(f"LLM REQUEST (text) #{i+1} BEGIN")
            self._w(p)
            self._w(f"LLM REQUEST (text) #{i+1} END")

    def on_llm_end(self, response, **kwargs: Any) -> None:
        try:
            gens = getattr(response, "generations", []) or []
            for i, gen_list in enumerate(gens):
                if not gen_list:
                    continue
                text = getattr(gen_list[0], "text", "")
                self._w(f"LLM RESPONSE (text) #{i+1} BEGIN")
                self._w(text)
                self._w(f"LLM RESPONSE (text) #{i+1} END")
        except Exception as e:
            self._w(f"LLM RESPONSE (text parse error): {e!r}")

