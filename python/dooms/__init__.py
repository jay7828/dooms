"""dooms.cpp from Python: image question answering (SmolVLM) and robot actions (SmolVLA) on Apple Silicon.

    import dooms
    llm = dooms.LLM("qwen2.5-0.5b-instruct-q4_k_m.gguf")          # any text GGUF llama.cpp loads
    print(llm.chat([{"role": "user", "content": "Name three colours."}], max_new_tokens=40))
    vlm = dooms.VLM.from_dir("models/smolvlm")
    print(vlm.generate(rgb_array, "Describe this image in one sentence."))
    vla = dooms.VLA.from_dir("models/smolvla")
    actions = vla.infer([cam1, cam2, cam3], "pick up the red cube", state)   # (chunk_size, action_dim) float32

Images are numpy arrays (or anything numpy can convert, e.g. a PIL image) of shape (height, width, 3), dtype uint8, RGB. Handles are not thread-safe; close() them
(or use `with`) to release the models. The shared library is found through $DOOMS_LIB, next to this package, or in <repo>/build.
"""
import codecs
import ctypes
import os
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import _lib

__all__ = ["DoomsError", "LLM", "VLM", "VLA", "version"]

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
