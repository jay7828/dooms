"""The `dooms` command (also `python -m dooms`).

    dooms serve --model FILE.gguf [--host 127.0.0.1] [--port 8080] [--n-ctx N] [--model-name NAME]
    dooms pull NAME [--dir DIR]          NAME: an alias (dooms pull --help), hf:OWNER/REPO/FILE.gguf[@REVISION], smolvlm, smolvla or whisper-*
    dooms list [--dir DIR]

Exit codes: 0 success, 1 the command failed (message on stderr), 2 usage error.
"""
import argparse
import os
import signal
import sys
from typing import List, Optional

from . import DoomsError, LLM, version


def _serve(args) -> int:
    from . import server  # noqa: PLC0415 - only this command needs it

    if not 0 <= args.port <= 65535:
        print("dooms serve: --port must be in [0, 65535]", file=sys.stderr)
        return 2
    if args.n_ctx < 0:
        print("dooms serve: --n-ctx must be >= 0", file=sys.stderr)
        return 2
    if not os.path.isfile(args.model):
        print(f"dooms serve: no such model file: {args.model}", file=sys.stderr)
        return 1
    name = args.model_name or os.path.splitext(os.path.basename(args.model))[0]
    try:
        llm = LLM(args.model, n_ctx=args.n_ctx, n_threads=args.n_threads, n_gpu_layers=args.n_gpu_layers)
    except DoomsError as e:
        print(f"dooms serve: cannot load {args.model}: {e}", file=sys.stderr)
        return 1
    try:
        httpd = server.make_server(llm, name, args.host, args.port, quiet=args.quiet)
    except OSError as e:
        print(f"dooms serve: cannot listen on {args.host}:{args.port}: {e} (is another server using the port? try --port)", file=sys.stderr)
        llm.close()
        return 1
    host, port = httpd.server_address[:2]
    print(f"dooms serve: {name} (context {llm.context_size} tokens) on http://{host}:{port}/v1  - Ctrl+C to stop", file=sys.stderr, flush=True)
    def stop(signum, frame):  # SIGTERM (service managers, kill) stops like Ctrl+C: the model is closed before the interpreter exits
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        llm.close()
    return 0


def _pull(args) -> int:
    from . import pull  # noqa: PLC0415

    try:
        path = pull.pull(args.name, args.dir)
    except pull.PullError as e:
        print(f"dooms pull: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("dooms pull: interrupted; run the same command again to resume", file=sys.stderr)
        return 1
    print(path)
    return 0


def _list(args) -> int:
    from . import pull  # noqa: PLC0415

    d = pull.models_dir(args.dir)
    rows = pull.list_models(args.dir)
    if not rows:
        print(f"no models in {d} (dooms pull NAME)", file=sys.stderr)
        return 0
    for kind, path, size in rows:
        print(f"{kind:9} {size / 1e6:10,.1f} MB  {path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    from .server import DEFAULT_HOST, DEFAULT_PORT  # noqa: PLC0415

    p = argparse.ArgumentParser(prog="dooms", description="dooms.cpp: run models on Apple Silicon.")
    p.add_argument("--version", action="version", version=f"dooms {version()}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    s = sub.add_parser("serve", help="serve a text GGUF model over an OpenAI-compatible HTTP API",
                       description="Serves one text GGUF model with OpenAI's /v1/chat/completions, /v1/completions and /v1/models endpoints.")
    s.add_argument("--model", required=True, help="the GGUF file")
    s.add_argument("--host", default=DEFAULT_HOST, help=f"address to listen on (default {DEFAULT_HOST}; 0.0.0.0 exposes the server to the network)")
    s.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default {DEFAULT_PORT}; 0 picks a free one)")
    s.add_argument("--n-ctx", type=int, default=0, help="context size in tokens, prompt + answer (default 0: the library's default)")
    s.add_argument("--model-name", default=None, help="the model id clients use (default: the file name without .gguf)")
    s.add_argument("--n-threads", type=int, default=0, help="CPU threads (default 0: the library's default)")
    s.add_argument("--n-gpu-layers", type=int, default=-1, help="layers on the GPU (default -1: all)")
    s.add_argument("--quiet", action="store_true", help="do not log each request on stderr")
    s.set_defaults(func=_serve)

    from . import pull  # noqa: PLC0415

    aliases = "; ".join(f"{k}: {a.note} ({a.repo})" for k, a in pull.ALIASES.items())
    default_dir = f"${pull.MODELS_ENV}, else {pull.DEFAULT_MODELS_DIR}"
    d = sub.add_parser("pull", help="download a model from Hugging Face and verify it",
                       description=f"Downloads a GGUF and checks it against Hugging Face's published hash (resumable). Aliases: {aliases}. smolvlm, smolvla, whisper-tiny, "
                                   "whisper-large-v3-turbo and kokoro download the original weights and convert them locally (source checkout with the converter environment only).")
    d.add_argument("name", help="an alias, hf:OWNER/REPO/FILE.gguf[@REVISION], smolvlm, smolvla, whisper-* or kokoro")
    d.add_argument("--dir", default=None, help=f"models directory (default {default_dir})")
    d.set_defaults(func=_pull)
    li = sub.add_parser("list", help="show what is in the models directory")
    li.add_argument("--dir", default=None, help=f"models directory (default {default_dir})")
    li.set_defaults(func=_list)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    return args.func(args)
