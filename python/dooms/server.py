"""An OpenAI-compatible HTTP server for one text language model (`dooms serve`), built on the standard library only.

    server = make_server(dooms.LLM("model.gguf"), "my-model", port=0)     # port 0: any free port; server.server_address has the real one
    server.serve_forever()

Endpoints: GET /health, GET /v1/models, GET /v1/models/{id}, POST /v1/chat/completions, POST /v1/completions (streaming with `stream: true`, Server-Sent Events in OpenAI's
chunk format, ending with `data: [DONE]`). Requests are validated as OpenAI validates them: anything this server cannot honour (n > 1, tools, logprobs, images, non-zero
penalties, ...) is a 400 with an OpenAI error object instead of being silently ignored. The model handle is not thread-safe, so one lock serializes every request that
touches it; HTTP handling itself is threaded (a client waiting for the lock does not block /health). A client that disconnects during a stream stops the generation.
"""
import json
import secrets
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import DoomsError, LLM, version

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
MAX_BODY_BYTES = 16 * 1024 * 1024        # largest request body accepted (a prompt is text; the context of any model fits in far less)
MAX_STOP_SEQUENCES = 4                   # OpenAI's limit on `stop`
TEMPERATURE_MAX = 2.0                    # OpenAI's documented range is [0, 2]
DEFAULT_TEMPERATURE = 1.0                # OpenAI's defaults when a request omits them
DEFAULT_TOP_P = 1.0
SEED_MODULUS = 2 ** 32                   # the C API's seed is uint32; larger or negative seeds are reduced modulo this
OWNED_BY = "dooms"
# chat roles the model's template understands; OpenAI's "developer" role is the newer name of "system"
CHAT_ROLES = {"system": "system", "developer": "system", "user": "user", "assistant": "assistant"}
TEXT_PART_SEPARATOR = "\n"               # a message whose content is a list of text parts becomes the parts joined by this


class ApiError(Exception):
    """An error answered with OpenAI's error object: {"error": {"message", "type", "param", "code"}}."""

    def __init__(self, status: int, message: str, type_: str = "invalid_request_error", param: Optional[str] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status, self.message, self.type, self.param, self.code = status, message, type_, param, code

    def body(self) -> Dict[str, Any]:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


def _bad(message: str, param: Optional[str] = None, code: Optional[str] = None) -> ApiError:
    return ApiError(HTTPStatus.BAD_REQUEST, message, param=param, code=code)


def _unsupported(param: str, what: str) -> ApiError:
    return _bad(f"{what} is not supported by this server (dooms serve runs one text model)", param=param, code="unsupported_parameter")


# ------------------------------------------------------------------------------------------------ request validation (no model needed)

def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _require_default(body: Dict[str, Any], key: str, allowed: Tuple, what: str) -> None:
    """Accepts `key` only when absent or set to a value that asks for OpenAI's default behaviour (which is what this server does)."""
    if key in body and body[key] not in allowed:
        raise _unsupported(key, what)


def _positive_int(body: Dict[str, Any], key: str) -> Optional[int]:
    v = body.get(key)
    if v is None:
        return None
    if not _is_int(v) or v < 1:
        raise _bad(f"'{key}' must be an integer >= 1, got {json.dumps(v)}", param=key, code="invalid_value")
    return v


def _sampling(body: Dict[str, Any]) -> Dict[str, Any]:
    temperature = body.get("temperature", DEFAULT_TEMPERATURE)
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE
    if not _is_number(temperature) or not 0 <= temperature <= TEMPERATURE_MAX:
        raise _bad(f"'temperature' must be a number in [0, {TEMPERATURE_MAX:g}], got {json.dumps(temperature)}", param="temperature", code="invalid_value")
    top_p = body.get("top_p", DEFAULT_TOP_P)
    if top_p is None:
        top_p = DEFAULT_TOP_P
    if not _is_number(top_p) or not 0 <= top_p <= 1:
        raise _bad(f"'top_p' must be a number in [0, 1], got {json.dumps(top_p)}", param="top_p", code="invalid_value")
    seed = body.get("seed")
    if seed is not None and not _is_int(seed):
        raise _bad(f"'seed' must be an integer, got {json.dumps(seed)}", param="seed", code="invalid_value")
    # top_p 0 keeps only the most likely token: greedy, like temperature 0 (the engine requires top_p in (0, 1])
    greedy = temperature == 0 or top_p == 0
    return {
        "temperature": 0.0 if greedy else float(temperature),
        "top_k": 0,                                   # OpenAI has no top-k: off
        "top_p": 1.0 if greedy else float(top_p),
        "repeat_penalty": 1.0,
        "seed": (secrets.randbits(32) if seed is None else seed % SEED_MODULUS),
    }


def _stops(body: Dict[str, Any]) -> List[str]:
    stop = body.get("stop")
    if stop is None:
        return []
    stops = [stop] if isinstance(stop, str) else stop
    if not isinstance(stops, list) or not all(isinstance(s, str) for s in stops):
        raise _bad("'stop' must be a string or a list of strings", param="stop", code="invalid_type")
    if len(stops) > MAX_STOP_SEQUENCES:
        raise _bad(f"'stop' may hold at most {MAX_STOP_SEQUENCES} sequences, got {len(stops)}", param="stop", code="invalid_value")
    if any(s == "" for s in stops):
        raise _bad("'stop' sequences must not be empty", param="stop", code="invalid_value")
    return stops


def _stream_flags(body: Dict[str, Any]) -> Tuple[bool, bool]:
    stream = body.get("stream", False)
    if stream is None:
        stream = False
    if not isinstance(stream, bool):
        raise _bad("'stream' must be a boolean", param="stream", code="invalid_type")
    options = body.get("stream_options")
    include_usage = False
    if options is not None:
        if not stream:
            raise _bad("'stream_options' is only allowed when 'stream' is true", param="stream_options", code="invalid_value")
        if not isinstance(options, dict):
            raise _bad("'stream_options' must be an object", param="stream_options", code="invalid_type")
        for k, v in options.items():
            if k != "include_usage":
                raise _bad(f"Unrecognized stream option: {k}", param=f"stream_options.{k}", code="unknown_parameter")
            if not isinstance(v, bool):
                raise _bad("'stream_options.include_usage' must be a boolean", param="stream_options.include_usage", code="invalid_type")
            include_usage = v
    return stream, include_usage


# keys accepted by each endpoint. The ones in _NO_EFFECT do not change the output (bookkeeping fields OpenAI also accepts); the rest are handled below.
_NO_EFFECT = {"user", "metadata", "store", "service_tier"}
_CHAT_KEYS = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "seed", "stop", "stream", "stream_options", "n",
              "frequency_penalty", "presence_penalty", "logprobs", "top_logprobs", "tools", "tool_choice", "functions", "function_call", "parallel_tool_calls",
              "response_format", "logit_bias", "modalities", "audio", "prediction", "reasoning_effort", "web_search_options"} | _NO_EFFECT
_COMPLETION_KEYS = {"model", "prompt", "max_tokens", "temperature", "top_p", "seed", "stop", "stream", "stream_options", "n", "best_of", "echo", "logprobs", "suffix",
                    "frequency_penalty", "presence_penalty", "logit_bias"} | _NO_EFFECT


def _check_common(body: Dict[str, Any], allowed_keys) -> None:
    for k in body:
        if k not in allowed_keys:
            raise _bad(f"Unrecognized request argument supplied: {k}", param=k, code="unknown_parameter")
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise _bad("'model' must be a string", param="model", code="invalid_type")
    _require_default(body, "n", (None, 1), "n > 1 (several choices per request)")
    for k in ("frequency_penalty", "presence_penalty"):
        v = body.get(k)
        if v is not None and not _is_number(v):
            raise _bad(f"'{k}' must be a number", param=k, code="invalid_type")
        if v not in (None, 0, 0.0):  # 0 is OpenAI's default and means no penalty; anything else would change the output
            raise _unsupported(k, f"a non-zero {k}")
    _require_default(body, "logit_bias", (None, {}), "logit_bias")
    if "store" in body and body["store"] not in (None, False, True):
        raise _bad("'store' must be a boolean", param="store", code="invalid_type")


def _message_text(m: Dict[str, Any], i: int) -> str:
    content = m.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for j, part in enumerate(content):
            if not isinstance(part, dict) or "type" not in part:
                raise _bad(f"messages[{i}].content[{j}] must be an object with a 'type'", param=f"messages[{i}].content[{j}]", code="invalid_type")
            if part["type"] != "text":
                raise _unsupported(f"messages[{i}].content[{j}].type", f"content of type '{part['type']}' (only text)")
            if not isinstance(part.get("text"), str):
                raise _bad(f"messages[{i}].content[{j}].text must be a string", param=f"messages[{i}].content[{j}].text", code="invalid_type")
            texts.append(part["text"])
        return TEXT_PART_SEPARATOR.join(texts)
    raise _bad(f"messages[{i}].content must be a string or a list of text parts", param=f"messages[{i}].content", code="invalid_type")


def parse_chat_request(body: Any) -> Dict[str, Any]:
    """Validates a /v1/chat/completions body; returns the normalized request or raises ApiError."""
    if not isinstance(body, dict):
        raise _bad("the request body must be a JSON object", code="invalid_type")
    _check_common(body, _CHAT_KEYS)
    _require_default(body, "logprobs", (None, False), "logprobs")
    _require_default(body, "top_logprobs", (None,), "top_logprobs")
    _require_default(body, "tools", (None, []), "tools (function calling)")
    _require_default(body, "functions", (None, []), "functions (function calling)")
    _require_default(body, "tool_choice", (None, "none"), "tool_choice")
    _require_default(body, "function_call", (None, "none"), "function_call")
    _require_default(body, "response_format", (None, {"type": "text"}), "response_format other than text")
    _require_default(body, "modalities", (None, ["text"]), "output modalities other than text")
    for k in ("audio", "prediction", "reasoning_effort", "web_search_options"):
        _require_default(body, k, (None,), k)
    if "parallel_tool_calls" in body and not isinstance(body["parallel_tool_calls"], (bool, type(None))):
        raise _bad("'parallel_tool_calls' must be a boolean", param="parallel_tool_calls", code="invalid_type")
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise _bad("'messages' must be a non-empty list", param="messages", code="invalid_value" if isinstance(msgs, list) else "invalid_type")
    messages = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            raise _bad(f"messages[{i}] must be an object", param=f"messages[{i}]", code="invalid_type")
        role = m.get("role")
        if role in ("tool", "function"):
            raise _unsupported(f"messages[{i}].role", f"the '{role}' role (function calling)")
        if role not in CHAT_ROLES:
            raise _bad(f"messages[{i}].role must be one of {sorted(CHAT_ROLES)}, got {json.dumps(role)}", param=f"messages[{i}].role", code="invalid_value")
        for k in ("tool_calls", "function_call", "audio"):
            if m.get(k) not in (None, []):
                raise _unsupported(f"messages[{i}].{k}", f"messages with {k}")
        messages.append({"role": CHAT_ROLES[role], "content": _message_text(m, i)})
    max_tokens = _positive_int(body, "max_tokens")
    max_completion_tokens = _positive_int(body, "max_completion_tokens")
    if max_tokens is not None and max_completion_tokens is not None and max_tokens != max_completion_tokens:
        raise _bad("'max_tokens' and 'max_completion_tokens' disagree; send only 'max_completion_tokens'", param="max_tokens", code="invalid_value")
    stream, include_usage = _stream_flags(body)
    return {"kind": "chat", "model": body.get("model"), "messages": messages, "max_tokens": max_completion_tokens or max_tokens, "sampling": _sampling(body),
            "stop": _stops(body), "stream": stream, "include_usage": include_usage}


def parse_completion_request(body: Any) -> Dict[str, Any]:
    """Validates a /v1/completions body; returns the normalized request or raises ApiError."""
    if not isinstance(body, dict):
        raise _bad("the request body must be a JSON object", code="invalid_type")
    _check_common(body, _COMPLETION_KEYS)
    _require_default(body, "best_of", (None, 1), "best_of > 1")
    _require_default(body, "echo", (None, False), "echo")
    _require_default(body, "logprobs", (None,), "logprobs")
    _require_default(body, "suffix", (None,), "suffix (insertion)")
    prompt = body.get("prompt")
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
        prompt = prompt[0]
    if isinstance(prompt, list):
        raise _unsupported("prompt", "a list of prompts or token ids (send one string)")
    if not isinstance(prompt, str):
        raise _bad("'prompt' must be a string", param="prompt", code="invalid_type")
    if prompt == "":
        raise _bad("'prompt' must not be empty", param="prompt", code="invalid_value")
    stream, include_usage = _stream_flags(body)
    return {"kind": "completion", "model": body.get("model"), "prompt": prompt, "max_tokens": _positive_int(body, "max_tokens"), "sampling": _sampling(body),
            "stop": _stops(body), "stream": stream, "include_usage": include_usage}


# ------------------------------------------------------------------------------------------------ the model behind a lock

def _from_dooms_error(e: DoomsError) -> ApiError:
    if e.code in ("InvalidArgument", "Unsupported"):
        return _bad(str(e), code="invalid_value")
    if e.code == "ResourceExhausted":
        return _bad(str(e), param="messages", code="context_length_exceeded")
    return ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), type_="server_error", code="internal_error")


class LlmService:
    """One model, one lock. Every method that touches the handle takes the lock; `run` holds it for the whole generation (including a stream's writes)."""

    def __init__(self, llm: LLM, model_name: str):
        self.llm = llm
        self.model_name = model_name
        self.created = int(time.time())
        self.lock = threading.Lock()
        self.last_generation = None  # Generation of the last finished request (tests read it to see a cancelled stream)

    def check_model(self, requested: Optional[str]) -> None:
        if requested is not None and requested != self.model_name:
            raise ApiError(HTTPStatus.NOT_FOUND, f"The model '{requested}' does not exist; this server serves '{self.model_name}'", param="model", code="model_not_found")

    def _budget(self, req: Dict[str, Any]) -> int:
        """Prompt length check and the token budget (OpenAI: without max_tokens the answer may use the rest of the context). Called with the lock held."""
        try:
            n_prompt = self.llm.count_chat_tokens(req["messages"]) if req["kind"] == "chat" else self.llm.count_tokens(req["prompt"])
        except DoomsError as e:
            raise _from_dooms_error(e) from None
        n_ctx = self.llm.context_size
        max_tokens = req["max_tokens"] if req["max_tokens"] is not None else n_ctx - n_prompt
        if max_tokens < 1 or n_prompt + max_tokens > n_ctx:
            where = "messages" if req["kind"] == "chat" else "prompt"
            raise _bad(f"This model's maximum context length is {n_ctx} tokens. However, you requested {n_prompt + max(max_tokens, 1)} tokens ({n_prompt} in the {where}, "
                       f"{max(max_tokens, 1)} in the completion). Please reduce the length of the {where} or the completion (or start the server with a larger --n-ctx).",
                       param=where, code="context_length_exceeded")
        return max_tokens

    def run(self, req: Dict[str, Any], on_start: Callable[[], None], on_text: Optional[Callable[[str], bool]]):
        """Generates the answer. on_start() runs once the request is accepted (budget checked), before the first token: the streaming handler sends its headers there.
        Returns (text, Generation). Raises ApiError for a request the model rejects."""
        with self.lock:
            max_tokens = self._budget(req)
            on_start()
            s = req["sampling"]
            kwargs = dict(temperature=s["temperature"], top_k=s["top_k"], top_p=s["top_p"], repeat_penalty=s["repeat_penalty"], seed=s["seed"], stop=req["stop"] or None,
                          on_text=on_text)
            try:
                if req["kind"] == "chat":
                    text = self.llm.chat(req["messages"], max_tokens, **kwargs)
                else:
                    text = self.llm.generate(req["prompt"], max_tokens, **kwargs)
            except DoomsError as e:
                self.last_generation = None
                raise _from_dooms_error(e) from None
            self.last_generation = self.llm.last_generation
            return text, self.last_generation


# ------------------------------------------------------------------------------------------------ HTTP

def _usage(g) -> Dict[str, int]:
    return {"prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens, "total_tokens": g.prompt_tokens + g.completion_tokens}


class _ClientGone(Exception):
    pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # keep-alive; streams use chunked transfer encoding
    server_version = "dooms/" + version()

    @property
    def service(self) -> LlmService:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, format, *args):  # noqa: A002 - signature of the base class
        if not getattr(self.server, "quiet", False):
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    # -------------------------------------------------------------------------------- responses
    def _send_json(self, status: int, obj: Dict[str, Any]) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, e: ApiError) -> None:
        self._send_json(e.status, e.body())

    def _chunk(self, data: bytes) -> None:
        try:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
        except OSError as e:  # BrokenPipe / ConnectionReset: the client went away
            raise _ClientGone() from e

    def _event(self, obj) -> None:
        payload = obj if isinstance(obj, str) else json.dumps(obj)
        self._chunk(b"data: " + payload.encode() + b"\n\n")

    # -------------------------------------------------------------------------------- routing
    def do_GET(self):  # noqa: N802 - http.server naming
        path = self.path.split("?", 1)[0].rstrip("/")
        svc = self.service
        model = {"id": svc.model_name, "object": "model", "created": svc.created, "owned_by": OWNED_BY}
        if path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok", "model": svc.model_name})
        elif path == "/v1/models":
            self._send_json(HTTPStatus.OK, {"object": "list", "data": [model]})
        elif path.startswith("/v1/models/"):
            try:
                svc.check_model(path[len("/v1/models/"):])
                self._send_json(HTTPStatus.OK, model)
            except ApiError as e:
                self._send_error(e)
        elif path in ("/v1/chat/completions", "/v1/completions"):
            self._send_error(ApiError(HTTPStatus.METHOD_NOT_ALLOWED, f"{path} takes POST, not GET", code="method_not_allowed"))
        else:
            self._send_error(ApiError(HTTPStatus.NOT_FOUND, f"Unknown request URL: GET {path}", code="unknown_url"))

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        parsers = {"/v1/chat/completions": parse_chat_request, "/v1/completions": parse_completion_request}
        try:
            body = self._read_json()
            if path not in parsers:
                raise ApiError(HTTPStatus.NOT_FOUND, f"Unknown request URL: POST {path}", code="unknown_url")
            req = parsers[path](body)
            self.service.check_model(req["model"])
        except ApiError as e:
            self._send_error(e)
            return
        if req["stream"]:
            self._stream(req)
        else:
            self._complete(req)

    def _read_json(self) -> Any:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            self.close_connection = True
            raise ApiError(HTTPStatus.LENGTH_REQUIRED, "chunked request bodies are not supported; send Content-Length", code="length_required")
        length = self.headers.get("Content-Length")
        try:
            n = int(length) if length is not None else 0
        except ValueError:
            self.close_connection = True
            raise _bad("Content-Length is not a number", code="invalid_header") from None
        if n < 0:
            self.close_connection = True
            raise _bad("Content-Length is negative", code="invalid_header")
        if n > MAX_BODY_BYTES:
            self.close_connection = True  # the body is not read: the connection cannot be reused
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"the request body has {n} bytes; the limit is {MAX_BODY_BYTES}", code="request_too_large")
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            raise _bad(f"We could not parse the JSON body of your request ({e}). The body must be a JSON object (Content-Type: application/json).",
                       code="invalid_json") from None

    # -------------------------------------------------------------------------------- generation
    def _base(self, req, object_name: str) -> Dict[str, Any]:
        prefix = "chatcmpl-" if req["kind"] == "chat" else "cmpl-"
        return {"id": prefix + secrets.token_hex(12), "object": object_name, "created": int(time.time()), "model": self.service.model_name,
                "system_fingerprint": "dooms-" + version()}

    def _complete(self, req) -> None:
        try:
            text, g = self.service.run(req, lambda: None, None)
        except ApiError as e:
            self._send_error(e)
            return
        if req["kind"] == "chat":
            out = self._base(req, "chat.completion")
            out["choices"] = [{"index": 0, "message": {"role": "assistant", "content": text, "refusal": None}, "logprobs": None, "finish_reason": g.finish_reason}]
        else:
            out = self._base(req, "text_completion")
            out["choices"] = [{"text": text, "index": 0, "logprobs": None, "finish_reason": g.finish_reason}]
        out["usage"] = _usage(g)
        try:
            self._send_json(HTTPStatus.OK, out)
        except OSError:  # the client left while the answer was being generated
            self.close_connection = True

    def _stream(self, req) -> None:
        chat = req["kind"] == "chat"
        base = self._base(req, "chat.completion.chunk" if chat else "text_completion")
        if req["include_usage"]:
            base["usage"] = None
        started = []

        def choice(piece: Optional[str], finish: Optional[str], first: bool = False) -> Dict[str, Any]:
            if chat:
                delta = {"role": "assistant", "content": ""} if first else ({} if piece is None else {"content": piece})
                return {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
            return {"text": piece or "", "index": 0, "logprobs": None, "finish_reason": finish}

        def start():
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            started.append(True)
            if chat:
                self._event(dict(base, choices=[choice(None, None, first=True)]))

        gone = []

        def on_text(piece: str) -> bool:
            try:
                self._event(dict(base, choices=[choice(piece, None)]))
                return True
            except _ClientGone:
                gone.append(True)
                return False  # stops the generation

        try:
            _, g = self.service.run(req, start, on_text)
        except _ClientGone:  # raised by start() (the role chunk)
            self.close_connection = True
            return
        except ApiError as e:
            if not started:
                self._send_error(e)
                return
            try:  # the headers are out: report in the stream, as OpenAI does
                self._event(e.body())
                self._chunk(b"")
            except _ClientGone:
                pass
            self.close_connection = True
            return
        if gone:
            self.close_connection = True
            return
        try:
            self._event(dict(base, choices=[choice(None, g.finish_reason)]))
            if req["include_usage"]:
                self._event(dict(base, choices=[], usage=_usage(g)))
            self._event("[DONE]")
            self._chunk(b"")  # end of the chunked body
        except _ClientGone:
            self.close_connection = True


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, service: LlmService, quiet: bool):
        self.service = service
        self.quiet = quiet
        super().__init__(address, Handler)


def make_server(llm: LLM, model_name: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, quiet: bool = False) -> ThreadingHTTPServer:
    """An HTTP server (not yet serving: call serve_forever()) for `llm` under the name `model_name`. port 0 picks a free port (see server_address)."""
    if not model_name:
        raise ValueError("model_name must not be empty")
    return _Server((host, port), LlmService(llm, model_name), quiet)
