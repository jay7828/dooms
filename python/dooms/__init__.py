"""dooms.cpp from Python: image question answering (SmolVLM) and robot actions (SmolVLA) on Apple Silicon.

    import dooms
    llm = dooms.LLM("qwen2.5-0.5b-instruct-q4_k_m.gguf")          # any text GGUF llama.cpp loads
    print(llm.chat([{"role": "user", "content": "Name three colours."}], max_new_tokens=40))
    vlm = dooms.VLM.from_dir("models/smolvlm")
    print(vlm.generate(rgb_array, "Describe this image in one sentence."))
    vla = dooms.VLA.from_dir("models/smolvla")
    actions = vla.infer([cam1, cam2, cam3], "pick up the red cube", state)   # (chunk_size, action_dim) float32
    runner = vla.runner(control_hz=30)                # asynchronous chunks for a control loop: wants_observation / submit / next_action

Images are numpy arrays (or anything numpy can convert, e.g. a PIL image) of shape (height, width, 3), dtype uint8, RGB. Handles are not thread-safe; close() them
(or use `with`) to release the models. The shared library is found through $DOOMS_LIB, next to this package, or in <repo>/build.
"""
import codecs
import ctypes
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import _lib

__all__ = ["DoomsError", "Generation", "LLM", "VLM", "VLA", "VLARunner", "RunnerStats", "aggregate_weight", "version"]

_lib_handle = None


def _library():
    global _lib_handle
    if _lib_handle is None:
        _lib_handle = _lib.load()
    return _lib_handle


def version() -> str:
    return _library().dooms_version().decode()


class DoomsError(RuntimeError):
    """A failed call; `code` is the dooms_code name ("NotFound", "InvalidArgument", ...)."""

    def __init__(self, code: int, message: str):
        self.code_value = code
        self.code = _lib.CODES.get(code, str(code))
        super().__init__(f"{self.code}: {message}")


def _check(code: int) -> None:
    if code != 0:
        raise DoomsError(code, _library().dooms_last_error().decode(errors="replace"))


def _image(array) -> "tuple[_lib.Image, np.ndarray]":
    a = np.asarray(array)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected a uint8 array of shape (height, width, 3), got {a.dtype} {a.shape}")
    a = np.ascontiguousarray(a)
    img = _lib.Image(a.shape[1], a.shape[0], a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return img, a  # keep `a` alive while the call runs


def _b(path: Optional[str]) -> Optional[bytes]:
    return None if path is None else os.fspath(path).encode()


def _exists(path: str) -> Optional[str]:
    return path if os.path.exists(path) else None


class _Handle:
    _close = None

    def __init__(self):
        self._h = ctypes.c_void_p()

    def close(self) -> None:
        if self._h:
            getattr(_library(), self._close)(self._h)
            self._h = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass

    def _require_open(self) -> None:
        if not self._h:
            raise ValueError("this model has been closed")




class VLM(_Handle):
    _close = "dooms_vlm_close"

    def __init__(self, vision, text, config, *, vision_units=None, n_threads=0):
        super().__init__()
        paths = _lib.VlmPaths(ctypes.sizeof(_lib.VlmPaths), _b(vision), _b(text), _b(config))
        opts = _lib.VlmOptions(ctypes.sizeof(_lib.VlmOptions), vision_units.encode() if vision_units else None, n_threads)
        _check(_library().dooms_vlm_open(ctypes.byref(paths), ctypes.byref(opts), ctypes.byref(self._h)))

    @classmethod
    def from_dir(cls, models_dir: str, **kwargs) -> "VLM":
        d = os.fspath(models_dir)
        return cls(os.path.join(d, "vision-fp16.mlpackage"), os.path.join(d, "text-f16.gguf"), os.path.join(d, "vlm-config.gguf"), **kwargs)

    def generate(self, image, prompt: str, max_new_tokens: int = 128, on_text: Optional[Callable[[str], Optional[bool]]] = None) -> str:
        """Greedy answer of at most max_new_tokens tokens. `on_text(piece)` (optional) is called with each new piece of text as it is generated (a character is never
        split: pieces are decoded incrementally); returning False stops the generation, and the text so far is returned. Exceptions raised by the callback stop the
        generation and are re-raised here."""
        self._require_open()
        if on_text is not None:
            return self._generate_stream(image, prompt, max_new_tokens, on_text)
        img, keep = _image(image)
        capacity = 4 * max_new_tokens + 64  # tokens are at most a few bytes of UTF-8 each
        while True:
            buf = ctypes.create_string_buffer(capacity)
            needed = ctypes.c_size_t()
            code = _library().dooms_vlm_generate(self._h, ctypes.byref(img), prompt.encode(), max_new_tokens, buf, capacity, ctypes.byref(needed))
            if code == 5 and needed.value > capacity:  # buffer too small: the answer is deterministic, so just retry with the size it reported
                capacity = needed.value
                continue
            _check(code)
            del keep
            return buf.value.decode(errors="replace")


    def _generate_stream(self, image, prompt, max_new_tokens, on_text) -> str:
        img, keep = _image(image)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        parts: List[str] = []
        failure: List[BaseException] = []

        def callback(piece, _user):
            try:
                text = decoder.decode(piece)
                if text:
                    parts.append(text)
                    if on_text(text) is False:
                        return 1
                return 0
            except BaseException as e:  # noqa: BLE001 - must not unwind through C frames
                failure.append(e)
                return 1

        cb = _lib.TextCallback(callback)  # keep a reference for the duration of the call
        _check(_library().dooms_vlm_generate_stream(self._h, ctypes.byref(img), prompt.encode(), max_new_tokens, cb, None))
        del keep
        if failure:
            raise failure[0]
        tail = decoder.decode(b"", final=True)
        if tail:
            parts.append(tail)
        return "".join(parts)

    def stream(self, image, prompt: str, max_new_tokens: int = 128) -> Iterator[str]:
        """Iterator over the pieces of the answer as they are generated. Breaking out of the loop (or closing the iterator) stops the generation. The model runs on a
        helper thread; do not use this handle from another thread until the iterator is exhausted or closed."""
        self._require_open()
        q: "queue.Queue" = queue.Queue()
        stop = threading.Event()
        done = object()

        def on_text(piece):
            q.put(piece)
            return not stop.is_set()

        def worker():
            try:
                self._generate_stream(image, prompt, max_new_tokens, on_text)
                q.put(done)
            except BaseException as e:  # noqa: BLE001 - delivered to the consumer
                q.put(e)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is done:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            t.join()


@dataclass(frozen=True)
class Generation:
    """How the last LLM call ended: exact token counts and why it stopped.

    finish_reason: "stop" (end-of-generation token or a stop string), "length" (max_new_tokens reached), "cancelled" (on_text returned False or raised), "none" (no call yet,
    or the last one failed). completion_tokens counts every sampled token, including a final end-of-generation token and the tokens of a matched stop string."""
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class LLM(_Handle):
    """A text language model from a GGUF file (any architecture and quantization llama.cpp loads), run on the Apple GPU.

        llm = dooms.LLM("model-q4_k_m.gguf", n_ctx=4096)
        llm.generate("The capital of France is", max_new_tokens=32)
        llm.chat([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], temperature=0.7, seed=1)
        for piece in llm.stream_chat([{"role": "user", "content": "Tell a story."}], max_new_tokens=200): print(piece, end="")

    temperature 0 (the default) is greedy and deterministic; otherwise tokens are drawn with a generator seeded by `seed`. Each call is a fresh conversation. `stop` is a list of
    strings that end the text before their first occurrence. `on_text(piece)` is called with each new piece (never half a character); returning False stops."""
    _close = "dooms_llm_close"

    def __init__(self, path, *, n_ctx: int = 0, n_threads: int = 0, n_gpu_layers: int = -1):
        super().__init__()
        opts = _lib.LlmOptions(ctypes.sizeof(_lib.LlmOptions), n_ctx, n_threads, n_gpu_layers)
        _check(_library().dooms_llm_open(_b(path), ctypes.byref(opts), ctypes.byref(self._h)))

    @property
    def context_size(self) -> int:
        self._require_open()
        return int(_library().dooms_llm_context_size(self._h))

    @staticmethod
    def _require(symbol: str) -> None:
        if not hasattr(_library(), symbol):
            raise OSError(f"the loaded libdooms_c.dylib has no {symbol} (it predates this Python package); rebuild it or point DOOMS_LIB at a newer one")

    @property
    def last_generation(self) -> Generation:
        """Token counts and finish reason of the last generate()/chat()/stream() call on this model."""
        self._require_open()
        self._require("dooms_llm_last_generation")
        info = _lib.LlmGenerationInfo(ctypes.sizeof(_lib.LlmGenerationInfo))
        _check(_library().dooms_llm_last_generation(self._h, ctypes.byref(info)))
        return Generation(int(info.prompt_tokens), int(info.completion_tokens), _lib.FINISH_REASONS.get(info.finish_reason, str(info.finish_reason)))

    def count_tokens(self, text: str, add_special: bool = True) -> int:
        """Number of tokens `text` becomes; add_special=True includes the BOS token the model asks for, as generate() tokenizes its prompt."""
        self._require_open()
        self._require("dooms_llm_count_tokens")
        n = ctypes.c_int32()
        _check(_library().dooms_llm_count_tokens(self._h, text.encode(), 1 if add_special else 0, ctypes.byref(n)))
        return int(n.value)

    def count_chat_tokens(self, messages) -> int:
        """Number of prompt tokens chat() feeds the model for `messages` (the chat template's output)."""
        self._require_open()
        self._require("dooms_llm_count_chat_tokens")
        msgs, n_msgs = self._messages(messages)
        n = ctypes.c_int32()
        _check(_library().dooms_llm_count_chat_tokens(self._h, msgs, n_msgs, ctypes.byref(n)))
        return int(n.value)

    @staticmethod
    def _sampling(temperature, top_k, top_p, repeat_penalty, seed):
        return _lib.LlmSampling(ctypes.sizeof(_lib.LlmSampling), temperature, top_k, top_p, repeat_penalty, seed)

    @staticmethod
    def _stops(stop):
        stop = list(stop or [])
        arr = (ctypes.c_char_p * max(len(stop), 1))(*[s.encode() for s in stop])
        return arr, len(stop)

    @staticmethod
    def _messages(messages):
        items = []
        for m in messages:
            role, content = (m["role"], m["content"]) if isinstance(m, dict) else (m[0], m[1])
            items.append(_lib.ChatMessage(role.encode(), content.encode()))
        if not items:
            raise ValueError("no messages")
        return (_lib.ChatMessage * len(items))(*items), len(items)

    def _call(self, kind, subject, max_new_tokens, temperature, top_k, top_p, repeat_penalty, seed, stop, on_text) -> str:
        self._require_open()
        lib = _library()
        samp = self._sampling(temperature, top_k, top_p, repeat_penalty, seed)
        stops, n_stop = self._stops(stop)
        if kind == "chat":
            msgs, n_msgs = self._messages(subject)
            head = (self._h, msgs, n_msgs, max_new_tokens)
            direct, streamed = lib.dooms_llm_chat, lib.dooms_llm_chat_stream
        else:
            head = (self._h, subject.encode(), max_new_tokens)
            direct, streamed = lib.dooms_llm_generate, lib.dooms_llm_generate_stream
        if on_text is None:
            capacity = 8 * max_new_tokens + 64
            while True:
                buf = ctypes.create_string_buffer(capacity)
                needed = ctypes.c_size_t()
                code = direct(*head, ctypes.byref(samp), stops, n_stop, buf, capacity, ctypes.byref(needed))
                if code == 5 and needed.value > capacity:  # buffer too small; the answer is deterministic for a fixed seed, so retry with the reported size
                    capacity = needed.value
                    continue
                _check(code)
                return buf.value.decode(errors="replace")
        parts: List[str] = []
        failure: List[BaseException] = []

        def callback(piece, _user):
            try:
                text = piece.decode(errors="replace")
                parts.append(text)
                if on_text(text) is False:
                    return 1
                return 0
            except BaseException as e:  # noqa: BLE001 - must not unwind through C frames
                failure.append(e)
                return 1

        cb = _lib.TextCallback(callback)
        _check(streamed(*head, ctypes.byref(samp), stops, n_stop, cb, None))
        if failure:
            raise failure[0]
        return "".join(parts)

    def generate(self, prompt: str, max_new_tokens: int = 128, *, temperature: float = 0.0, top_k: int = 40, top_p: float = 0.95, repeat_penalty: float = 1.0,
                 seed: int = 0, stop: Optional[Sequence[str]] = None, on_text: Optional[Callable[[str], Optional[bool]]] = None) -> str:
        return self._call("generate", prompt, max_new_tokens, temperature, top_k, top_p, repeat_penalty, seed, stop, on_text)

    def chat(self, messages, max_new_tokens: int = 128, *, temperature: float = 0.0, top_k: int = 40, top_p: float = 0.95, repeat_penalty: float = 1.0,
             seed: int = 0, stop: Optional[Sequence[str]] = None, on_text: Optional[Callable[[str], Optional[bool]]] = None) -> str:
        """`messages`: a list of {"role": ..., "content": ...} dicts (or (role, content) pairs), formatted with the model's own chat template."""
        return self._call("chat", messages, max_new_tokens, temperature, top_k, top_p, repeat_penalty, seed, stop, on_text)

    def _iterate(self, call) -> Iterator[str]:
        q: "queue.Queue" = queue.Queue()
        stop_flag = threading.Event()
        done = object()

        def on_text(piece):
            q.put(piece)
            return not stop_flag.is_set()

        def worker():
            try:
                call(on_text)
                q.put(done)
            except BaseException as e:  # noqa: BLE001 - delivered to the consumer
                q.put(e)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is done:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop_flag.set()
            t.join()

    def stream(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> Iterator[str]:
        """Iterator over the pieces of generate() as they are produced. The model runs on a helper thread; do not use this handle elsewhere until the iterator is done."""
        self._require_open()
        return self._iterate(lambda cb: self.generate(prompt, max_new_tokens, on_text=cb, **kwargs))

    def stream_chat(self, messages, max_new_tokens: int = 128, **kwargs) -> Iterator[str]:
        self._require_open()
        return self._iterate(lambda cb: self.chat(messages, max_new_tokens, on_text=cb, **kwargs))


class VLA(_Handle):
    _close = "dooms_vla_close"

    def __init__(self, vision, prefix, expert, host, vocab, *, units=None, seed=0):
        super().__init__()
        paths = _lib.VlaPaths(ctypes.sizeof(_lib.VlaPaths), _b(vision), _b(prefix), _b(expert), _b(host), _b(vocab))
        opts = _lib.VlaOptions(ctypes.sizeof(_lib.VlaOptions), units.encode() if units else None, seed)
        _check(_library().dooms_vla_open(ctypes.byref(paths), ctypes.byref(opts), ctypes.byref(self._h)))
        info = _lib.VlaInfo()
        _check(_library().dooms_vla_info_get(self._h, ctypes.byref(info)))
        self.n_cameras, self.chunk_size, self.action_dim, self.state_dim, self.max_task_tokens = info.n_cameras, info.chunk_size, info.action_dim, info.state_dim, info.max_task_tokens

    @classmethod
    def from_dir(cls, models_dir: str, **kwargs) -> "VLA":
        d = os.fspath(models_dir)
        j = lambda n: os.path.join(d, n)  # noqa: E731
        return cls(j("vision-b1-fp16.mlpackage"), j("prefix-fp16.mlpackage"), j("expert-fp16.mlpackage"), j("host.gguf"), j("vocab.gguf"), **kwargs)

    def infer(self, cameras: Sequence, task: str, state: Sequence[float], noise: Optional[np.ndarray] = None) -> np.ndarray:
        """cameras: n_cameras uint8 RGB arrays; state: normalized robot state (<= state_dim values); noise: optional (chunk_size, action_dim) start of the flow for reproducible runs
        (default: random, seeded by the constructor). Returns (chunk_size, action_dim) float32."""
        self._require_open()
        if len(cameras) != self.n_cameras:
            raise ValueError(f"expected {self.n_cameras} camera images, got {len(cameras)}")
        prepared = [_image(c) for c in cameras]
        arr = (_lib.Image * len(prepared))(*[p[0] for p in prepared])
        st = np.ascontiguousarray(np.asarray(state, dtype=np.float32).reshape(-1))
        out = np.empty((self.chunk_size, self.action_dim), dtype=np.float32)
        nz = None
        if noise is not None:
            nz = np.ascontiguousarray(noise, dtype=np.float32)
            if nz.shape != out.shape:
                raise ValueError(f"noise must have shape {out.shape}, got {nz.shape}")
        _check(_library().dooms_vla_infer(self._h, arr, len(prepared), task.encode(), st.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), st.size,
                                          nz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if nz is not None else None, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.size))
        return out

    def runner(self, *, control_hz: Optional[float] = None, chunk_size_threshold: Optional[float] = None, aggregate=None) -> "VLARunner":
        """An asynchronous runner for a control loop (see VLARunner). While it is open, infer() on this handle raises: the runner's thread owns the model."""
        self._require_open()
        return VLARunner(self, control_hz=control_hz, chunk_size_threshold=chunk_size_threshold, aggregate=aggregate)


@dataclass(frozen=True)
class RunnerStats:
    actions_returned: int
    waiting_calls: int       # next_action() without an action before the first one
    stalled_calls: int       # ... after it: the loop ran out of actions
    late_skipped: int        # actions skipped because the loop called late
    stale_dropped: int       # rows of arrived chunks whose step had already run
    blended: int             # rows aggregated with a queued action
    chunks_received: int
    chunks_stale: int
    inferences_started: int
    inferences_failed: int
    queued: int
    in_flight: bool
    last_inference_ms: float
    last_error: Optional[str]  # "Code: message" of the last failed inference


def aggregate_weight(name: str) -> float:
    """lerobot's aggregate function name ("weighted_average", "latest_only", "average", "conservative") -> weight of the new chunk."""
    w = ctypes.c_double()
    _check(_library().dooms_vla_aggregate_weight(name.encode(), ctypes.byref(w)))
    return w.value


class VLARunner(_Handle):
    """Asynchronous action chunks for a robot control loop (lerobot's async inference, in-process). The next chunk is computed on a worker thread from a fresh
    observation while the robot executes the current one, and spliced in without a gap: actions whose step has already run are dropped, the overlap with the
    queued plan is blended ((1 - new_weight) * queued + new_weight * new), the rest appended. Nothing ever calls back into Python from the worker thread.

        with vla.runner(control_hz=30) as r:
            while running:
                now = time.monotonic()
                if r.wants_observation():
                    r.submit(read_cameras(), "pick up the cube", read_state(), now)
                action = r.next_action(now)        # None: no action this step (start-up or a stall): hold the last command
                if action is not None:
                    send(action[:6])
                sleep_until_next_tick()

    control_hz: control steps per second (default 30). chunk_size_threshold: ask for an observation when queued / chunk_size <= this (default 0.5; 0 is
    synchronous). aggregate: a lerobot name or the new chunk's weight in [0, 1] (default "weighted_average" = 0.7; "latest_only" = 1 replaces). Defaults come
    from the library. Times are seconds on time.monotonic() unless given. Use one runner from one thread."""
    _close = "dooms_vla_runner_close"

    def __init__(self, vla: "VLA", *, control_hz: Optional[float] = None, chunk_size_threshold: Optional[float] = None, aggregate=None):
        super().__init__()
        vla._require_open()
        opts = _lib.VlaRunnerOptions()
        _check(_library().dooms_vla_runner_options_init(ctypes.byref(opts)))
        if control_hz is not None:
            opts.control_hz = float(control_hz)
        if chunk_size_threshold is not None:
            opts.chunk_size_threshold = float(chunk_size_threshold)
        if isinstance(aggregate, str):
            opts.new_weight = aggregate_weight(aggregate)
        elif aggregate is not None:
            opts.new_weight = float(aggregate)
        _check(_library().dooms_vla_runner_open(vla._h, ctypes.byref(opts), ctypes.byref(self._h)))
        self.n_cameras, self.action_dim, self.state_dim = vla.n_cameras, vla.action_dim, vla.state_dim
        self.control_hz, self.chunk_size_threshold, self.new_weight = opts.control_hz, opts.chunk_size_threshold, opts.new_weight
        self.step = -1     # control step of the last action returned by next_action (-1: none)
        self.skipped = 0   # actions skipped in the last next_action call because it came late
        self._action = np.empty(self.action_dim, dtype=np.float32)
        self._info = _lib.VlaActionInfo(ctypes.sizeof(_lib.VlaActionInfo))

    def _require_open(self) -> None:
        if not self._h:
            raise ValueError("this runner has been closed")

    def wants_observation(self) -> bool:
        """True when submit() would start an inference. Cheap: capture cameras only when this is True."""
        self._require_open()
        w = ctypes.c_int32()
        _check(_library().dooms_vla_runner_wants_observation(self._h, ctypes.byref(w)))
        return bool(w.value)

    def submit(self, cameras: Sequence, task: str, state: Sequence[float], timestamp: Optional[float] = None) -> bool:
        """The observation taken at `timestamp` (default: now). Returns whether an inference started (an unneeded observation is dropped, never queued)."""
        self._require_open()
        if len(cameras) != self.n_cameras:
            raise ValueError(f"expected {self.n_cameras} camera images, got {len(cameras)}")
        t = time.monotonic() if timestamp is None else float(timestamp)
        prepared = [_image(c) for c in cameras]
        arr = (_lib.Image * len(prepared))(*[p[0] for p in prepared])
        st = np.ascontiguousarray(np.asarray(state, dtype=np.float32).reshape(-1))
        started = ctypes.c_int32()
        _check(_library().dooms_vla_runner_submit(self._h, arr, len(prepared), task.encode(), st.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), st.size, t,
                                                  ctypes.byref(started)))
        return bool(started.value)

    def next_action(self, now: Optional[float] = None) -> Optional[np.ndarray]:
        """The (action_dim,) float32 action for the control step at `now` (default: now; never earlier than the previous call), or None when there is none
        (before the first chunk, or the queue ran dry). Never waits for an inference. Sets self.step and self.skipped."""
        self._require_open()
        t = time.monotonic() if now is None else float(now)
        _check(_library().dooms_vla_runner_next_action(self._h, t, self._action.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self._action.size, ctypes.byref(self._info)))
        self.skipped = self._info.skipped
        if not self._info.available:
            return None
        self.step = self._info.step
        return self._action.copy()

    def wait_idle(self, timeout: float = 600.0) -> bool:
        """Waits up to `timeout` seconds for the inference in flight (start-up: wait for the first chunk). Returns whether none is in flight."""
        self._require_open()
        idle = ctypes.c_int32()
        _check(_library().dooms_vla_runner_wait_idle(self._h, float(timeout), ctypes.byref(idle)))
        return bool(idle.value)

    def stats(self) -> RunnerStats:
        self._require_open()
        s = _lib.VlaRunnerStats(ctypes.sizeof(_lib.VlaRunnerStats))
        _check(_library().dooms_vla_runner_stats_get(self._h, ctypes.byref(s)))
        return RunnerStats(s.actions_returned, s.waiting_calls, s.stalled_calls, s.late_skipped, s.stale_dropped, s.blended, s.chunks_received, s.chunks_stale,
                           s.inferences_started, s.inferences_failed, s.queued, bool(s.in_flight), s.last_inference_ms,
                           s.last_error.decode(errors="replace") if s.last_error_code != 0 else None)
