"""ctypes declarations of include/dooms/dooms_c.h and the library lookup. Nothing here is public API; use the classes in `dooms`."""
import ctypes
import os
import sys
from ctypes import POINTER, c_char_p, c_double, c_float, c_int32, c_int64, c_size_t, c_uint8, c_uint32, c_uint64, c_void_p

C_API_VERSION = 1

CODES = {0: "Ok", 1: "InvalidArgument", 2: "NotFound", 3: "DataLoss", 4: "Unsupported", 5: "ResourceExhausted", 6: "IoError", 7: "Internal"}


class Image(ctypes.Structure):
    _fields_ = [("width", c_int32), ("height", c_int32), ("rgb", POINTER(c_uint8))]




TextCallback = ctypes.CFUNCTYPE(c_int32, c_char_p, c_void_p)


class VlmPaths(ctypes.Structure):
    _fields_ = [("struct_size", c_size_t), ("vision", c_char_p), ("text", c_char_p), ("config", c_char_p)]


class VlmOptions(ctypes.Structure):
    _fields_ = [("struct_size", c_size_t), ("vision_units", c_char_p), ("n_threads", c_int32)]


class LlmOptions(ctypes.Structure):
    _fields_ = [("struct_size", c_size_t), ("n_ctx", c_int32), ("n_threads", c_int32), ("n_gpu_layers", c_int32)]


class LlmSampling(ctypes.Structure):
    _fields_ = [("struct_size", c_size_t), ("temperature", c_float), ("top_k", c_int32), ("top_p", c_float), ("repeat_penalty", c_float), ("seed", c_uint32)]


class ChatMessage(ctypes.Structure):
    _fields_ = [("role", c_char_p), ("content", c_char_p)]


class LlmGenerationInfo(ctypes.Structure):
    _fields_ = [("struct_size", c_size_t), ("prompt_tokens", c_int32), ("completion_tokens", c_int32), ("finish_reason", c_int32)]


# dooms_finish_reason values -> names (DOOMS_FINISH_NONE ... DOOMS_FINISH_CANCELLED)
FINISH_REASONS = {0: "none", 1: "stop", 2: "length", 3: "cancelled"}


class VlaPaths(ctypes.Structure):
    _fields_ = [("struct_size", c_size_t), ("vision", c_char_p), ("prefix", c_char_p), ("expert", c_char_p), ("host", c_char_p), ("vocab", c_char_p)]


class VlaOptions(ctypes.Structure):
    _fields_ = [("struct_size", c_size_t), ("units", c_char_p), ("seed", c_uint64)]


class VlaRunnerOptions(ctypes.Structure):
    _fields_ = [("struct_size", c_size_t), ("control_hz", c_double), ("chunk_size_threshold", c_double), ("new_weight", c_double)]


class VlaActionInfo(ctypes.Structure):
    _fields_ = [("struct_size", c_size_t), ("available", c_int32), ("skipped", c_int32), ("step", c_int64)]


class VlaRunnerStats(ctypes.Structure):
    _fields_ = [("struct_size", c_size_t), ("actions_returned", c_int64), ("waiting_calls", c_int64), ("stalled_calls", c_int64), ("late_skipped", c_int64),
                ("stale_dropped", c_int64), ("blended", c_int64), ("chunks_received", c_int64), ("chunks_stale", c_int64), ("inferences_started", c_int64),
                ("inferences_failed", c_int64), ("queued", c_int64), ("in_flight", c_int32), ("last_error_code", c_int32), ("last_inference_ms", c_double),
                ("last_error", c_char_p)]


class VlaInfo(ctypes.Structure):
    _fields_ = [("n_cameras", c_int32), ("chunk_size", c_int32), ("action_dim", c_int32), ("state_dim", c_int32), ("max_task_tokens", c_int32)]


def _candidates():
    env = os.environ.get("DOOMS_LIB")
    if env:
        yield env
    here = os.path.dirname(os.path.abspath(__file__))
    yield os.path.join(here, "lib", "libdooms_c.dylib")  # a wheel bundles the library and its dependencies here
    yield os.path.join(here, "libdooms_c.dylib")
    root = os.path.abspath(os.path.join(here, "..", ".."))
    yield os.path.join(root, "build", "libdooms_c.dylib")


def load():
    if sys.platform != "darwin":
        raise OSError("dooms.cpp runs on Apple Silicon macOS only")
    errors = []
    for path in _candidates():
        if not os.path.exists(path):
            errors.append(f"{path}: not found")
            continue
        try:
            lib = ctypes.CDLL(path)
        except OSError as e:
            errors.append(f"{path}: {e}")
            continue
        _declare(lib)
        if lib.dooms_c_api_version() != C_API_VERSION:
            raise OSError(f"{path} implements C API version {lib.dooms_c_api_version()}, this Python package needs {C_API_VERSION}")
        return lib
    raise OSError("libdooms_c.dylib not found; build the project or set DOOMS_LIB. Tried: " + "; ".join(errors))


def _declare(lib):
    code = c_int32
    lib.dooms_version.restype = c_char_p
    lib.dooms_c_api_version.restype = c_int32
    lib.dooms_last_error.restype = c_char_p
    lib.dooms_code_name.restype = c_char_p
    lib.dooms_code_name.argtypes = [c_int32]
    lib.dooms_vlm_generate_stream.restype = code
    lib.dooms_vlm_generate_stream.argtypes = [c_void_p, POINTER(Image), c_char_p, c_int32, TextCallback, c_void_p]
    lib.dooms_vlm_open.restype = code
    lib.dooms_vlm_open.argtypes = [POINTER(VlmPaths), POINTER(VlmOptions), POINTER(c_void_p)]
    lib.dooms_vlm_close.argtypes = [c_void_p]
    lib.dooms_llm_open.restype = code
    lib.dooms_llm_open.argtypes = [c_char_p, POINTER(LlmOptions), POINTER(c_void_p)]
    lib.dooms_llm_close.argtypes = [c_void_p]
    lib.dooms_llm_context_size.restype = c_int32
    lib.dooms_llm_context_size.argtypes = [c_void_p]
    lib.dooms_llm_generate.restype = code
    lib.dooms_llm_generate.argtypes = [c_void_p, c_char_p, c_int32, POINTER(LlmSampling), POINTER(c_char_p), c_int32, ctypes.c_char_p, c_size_t, POINTER(c_size_t)]
    lib.dooms_llm_generate_stream.restype = code
    lib.dooms_llm_generate_stream.argtypes = [c_void_p, c_char_p, c_int32, POINTER(LlmSampling), POINTER(c_char_p), c_int32, TextCallback, c_void_p]
    lib.dooms_llm_chat.restype = code
    lib.dooms_llm_chat.argtypes = [c_void_p, POINTER(ChatMessage), c_int32, c_int32, POINTER(LlmSampling), POINTER(c_char_p), c_int32, ctypes.c_char_p, c_size_t, POINTER(c_size_t)]
    lib.dooms_llm_chat_stream.restype = code
    lib.dooms_llm_chat_stream.argtypes = [c_void_p, POINTER(ChatMessage), c_int32, c_int32, POINTER(LlmSampling), POINTER(c_char_p), c_int32, TextCallback, c_void_p]
    # appended to C API version 1 after its first release: a library built before then lacks them, and only the calls that need them fail (LLM._require)
    if hasattr(lib, "dooms_llm_last_generation"):
        lib.dooms_llm_last_generation.restype = code
        lib.dooms_llm_last_generation.argtypes = [c_void_p, POINTER(LlmGenerationInfo)]
        lib.dooms_llm_count_tokens.restype = code
        lib.dooms_llm_count_tokens.argtypes = [c_void_p, c_char_p, c_int32, POINTER(c_int32)]
        lib.dooms_llm_count_chat_tokens.restype = code
        lib.dooms_llm_count_chat_tokens.argtypes = [c_void_p, POINTER(ChatMessage), c_int32, POINTER(c_int32)]
    lib.dooms_vlm_generate.restype = code
    lib.dooms_vlm_generate.argtypes = [c_void_p, POINTER(Image), c_char_p, c_int32, ctypes.c_char_p, c_size_t, POINTER(c_size_t)]
    lib.dooms_vla_open.restype = code
    lib.dooms_vla_open.argtypes = [POINTER(VlaPaths), POINTER(VlaOptions), POINTER(c_void_p)]
    lib.dooms_vla_close.argtypes = [c_void_p]
    lib.dooms_vla_info_get.restype = code
    lib.dooms_vla_info_get.argtypes = [c_void_p, POINTER(VlaInfo)]
    lib.dooms_vla_infer.restype = code
    lib.dooms_vla_infer.argtypes = [c_void_p, POINTER(Image), c_int32, c_char_p, POINTER(c_float), c_int32, POINTER(c_float), POINTER(c_float), c_size_t]
    # asynchronous chunks: every call returns promptly except wait_idle; ctypes releases the GIL during each call and the runner's thread never calls into Python
    lib.dooms_vla_aggregate_weight.restype = code
    lib.dooms_vla_aggregate_weight.argtypes = [c_char_p, POINTER(c_double)]
    lib.dooms_vla_runner_options_init.restype = code
    lib.dooms_vla_runner_options_init.argtypes = [POINTER(VlaRunnerOptions)]
    lib.dooms_vla_runner_open.restype = code
    lib.dooms_vla_runner_open.argtypes = [c_void_p, POINTER(VlaRunnerOptions), POINTER(c_void_p)]
    lib.dooms_vla_runner_close.argtypes = [c_void_p]
    lib.dooms_vla_runner_wants_observation.restype = code
    lib.dooms_vla_runner_wants_observation.argtypes = [c_void_p, POINTER(c_int32)]
    lib.dooms_vla_runner_submit.restype = code
    lib.dooms_vla_runner_submit.argtypes = [c_void_p, POINTER(Image), c_int32, c_char_p, POINTER(c_float), c_int32, c_double, POINTER(c_int32)]
    lib.dooms_vla_runner_next_action.restype = code
    lib.dooms_vla_runner_next_action.argtypes = [c_void_p, c_double, POINTER(c_float), c_size_t, POINTER(VlaActionInfo)]
    lib.dooms_vla_runner_wait_idle.restype = code
    lib.dooms_vla_runner_wait_idle.argtypes = [c_void_p, c_double, POINTER(c_int32)]
    lib.dooms_vla_runner_stats_get.restype = code
    lib.dooms_vla_runner_stats_get.argtypes = [c_void_p, POINTER(VlaRunnerStats)]
