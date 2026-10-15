"""Tests of `dooms pull` / `dooms list` (python/dooms/pull.py) against a local stand-in for the Hugging Face API and its CDN (no network): verification by SHA-256
and git blob SHA-1, resume with Range, servers that ignore Range or break off, checksum mismatch, atomic rename, error messages, token handling across redirects,
the conversion preconditions, the command line.

Opt-in: DOOMS_TEST_PULL=1 downloads a real official GGUF (the qwen2.5-0.5b alias, ~490 MB) into a temporary directory, verifies it and loads it.
"""
import hashlib
import http.server
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.parse
from contextlib import redirect_stderr
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dooms import pull  # noqa: E402

PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
COMMIT = "0123456789abcdef0123456789abcdef01234567"
REPO = "owner/model-GGUF"


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeHub:
    """A tiny Hugging Face: /api/models/... (revision, paths-info, tree with pagination) and /{repo}/resolve/... redirecting to /cdn/... on another host name
    (localhost vs 127.0.0.1), which serves bytes with Range support. Knobs change one behaviour per test."""

    def __init__(self):
        self.files = {  # path -> bytes; LFS files are announced with their SHA-256, the others with the git blob SHA-1
            "model.gguf": bytes((i * 7 + 3) % 251 for i in range(300_000)),
            "config.json": b'{"a": 1}\n',
            "sub/tokenizer.json": b'{"vocab": []}\n',
        }
        self.lfs = {"model.gguf"}
        self.ignore_range = False
        self.corrupt = False
        self.extra_bytes = 0
        self.break_after = None       # bytes of the body to send before dropping the connection
        self.gated = False
        self.page_size = 2            # tree entries per page (forces pagination)
        self.extra_tree_paths = []    # hostile paths to list in the tree
        self.log = []                 # (method, path, headers) of every request
        hub = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj, headers=None):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def _entry(self, path):
                data = hub.files[path]
                e = {"type": "file", "oid": blob_sha1(data), "size": len(data), "path": path}
                if path in hub.lfs:
                    e["lfs"] = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data), "pointerSize": 134}
                return e

            def do_GET(self):  # noqa: N802
                hub.log.append(("GET", self.path, dict(self.headers)))
                url = urllib.parse.urlsplit(self.path)
                path = urllib.parse.unquote(url.path)
                if hub.gated and path.startswith(("/api/", f"/{REPO}/")):
                    return self._json(401, {"error": "Access to model owner/model-GGUF is restricted."})
                if path.startswith(f"/api/models/{REPO}/revision/"):
                    rev = path.rsplit("/", 1)[1]
                    return self._json(200, {"id": REPO, "sha": COMMIT}) if rev in ("main", COMMIT) else self._json(404, {"error": f"Invalid rev id: {rev}"})
                if path == f"/api/models/{REPO}/tree/{COMMIT}":
                    entries = [self._entry(p) for p in sorted(hub.files)] + [{"type": "file", "oid": "0" * 40, "size": 1, "path": p} for p in hub.extra_tree_paths]
                    entries.insert(0, {"type": "directory", "oid": "x", "size": 0, "path": "sub"})
                    page = int(urllib.parse.parse_qs(url.query).get("page", ["0"])[0])
                    chunk = entries[page * hub.page_size:(page + 1) * hub.page_size]
                    more = (page + 1) * hub.page_size < len(entries)
                    link = {"Link": f'<{f"/api/models/{REPO}/tree/{COMMIT}?recursive=true&page={page + 1}"}>; rel="next"'} if more else {}
                    return self._json(200, chunk, link)
                prefix = f"/{REPO}/resolve/{COMMIT}/"
                if path.startswith(prefix):
                    name = path[len(prefix):]
                    if name not in hub.files:
                        return self._json(404, {"error": "Entry not found"})
                    self.send_response(302)
                    self.send_header("Location", f"http://localhost:{hub.port}/cdn/{urllib.parse.quote(name)}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return None
                if path.startswith("/cdn/"):
                    return self._serve_bytes(path[len("/cdn/"):])
                return self._json(404, {"error": "Repository not found"})

            def _serve_bytes(self, name):
                data = hub.files[name]
                if hub.corrupt:
                    data = bytes([data[0] ^ 1]) + data[1:]
                data += b"x" * hub.extra_bytes
                start = 0
                rng = self.headers.get("Range")
                if rng and not hub.ignore_range:
                    start = int(rng[len("bytes="):].split("-")[0])
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if hub.break_after is not None:
                    self.wfile.write(body[:hub.break_after])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                hub.log.append(("POST", self.path, dict(self.headers)))
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if hub.gated:
                    return self._json(401, {"error": "restricted"})
                if urllib.parse.unquote(self.path) == f"/api/models/{REPO}/paths-info/{COMMIT}":
                    return self._json(200, [self._entry(p) for p in body["paths"] if p in hub.files])
                return self._json(401, {"error": "Invalid username or password."})  # what Hugging Face answers for a missing repository

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.endpoint = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def cdn_requests(self):
        return [h for m, p, h in self.log if p.startswith("/cdn/")]


class PullTest(unittest.TestCase):
    def setUp(self):
        self.hub = FakeHub()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.env = mock.patch.dict(os.environ, {pull.HF_ENDPOINT_ENV: self.hub.endpoint, pull.MODELS_ENV: self.dir})
        self.env.start()
        os.environ.pop(pull.HF_TOKEN_ENV, None)
        self.no_wait = mock.patch.object(pull, "RETRY_WAIT_SECONDS", 0)  # retries happen, without the pauses
        self.no_wait.start()

    def tearDown(self):
        self.no_wait.stop()
        self.env.stop()
        self.hub.close()
        self.tmp.cleanup()

    def run_pull(self, name, **kw):
        with redirect_stderr(io.StringIO()) as err:
            path = pull.pull(name, progress=kw.get("progress"))
        return path, err.getvalue()

    def expect_error(self, name, *fragments):
        with self.assertRaises(pull.PullError) as cm, redirect_stderr(io.StringIO()):
            pull.pull(name, progress=None)
        for f in fragments:
            self.assertIn(f, str(cm.exception))
        return str(cm.exception)

    @property
    def dest(self):
        return os.path.join(self.dir, "model.gguf")

    def test_download_verifies_and_renames(self):
        seen = []

        def progress(name, done, total):  # while bytes arrive, only the .part file exists
            seen.append((done, total, os.path.exists(self.dest), os.path.exists(self.dest + pull.PART_SUFFIX)))

        path, err = self.run_pull(f"hf:{REPO}/model.gguf", progress=progress)
        self.assertEqual(path, self.dest)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), self.hub.files["model.gguf"])
        self.assertFalse(os.path.exists(self.dest + pull.PART_SUFFIX))
        self.assertIn("verified (sha256", err)
        self.assertTrue(seen and all(not final and part for _, _, final, part in seen))
        self.assertEqual(seen[-1][:2], (len(self.hub.files["model.gguf"]),) * 2)
        # a second pull checks the file and downloads nothing
        n = len(self.hub.cdn_requests())
        _, err = self.run_pull(f"hf:{REPO}/model.gguf")
        self.assertIn("already present", err)
        self.assertEqual(len(self.hub.cdn_requests()), n)

    def test_revision_pinning_and_parsing(self):
        self.run_pull(f"hf:{REPO}/model.gguf@{COMMIT}")
        self.assertFalse(any("/revision/" in p for _, p, _ in self.hub.log))  # a commit is used as is
        self.assertTrue(any(f"/resolve/{COMMIT}/" in p for _, p, _ in self.hub.log))
        self.assertEqual(pull.parse_hf("hf:a/b/c/d.gguf@v1"), ("a/b", "c/d.gguf", "v1"))
        self.assertEqual(pull.parse_hf("hf:a/b/d.gguf"), ("a/b", "d.gguf", "main"))
        for bad in ("hf:a/b", "hf:a//c.gguf", "hf:a/b/c.gguf@", "hf:a/b/c.bin"):
            with self.subTest(bad=bad):
                self.assertRaises(pull.PullError, pull.parse_hf, bad)
        self.expect_error("nothing-like-this", "unknown model", "qwen2.5-0.5b", "hf:OWNER/REPO")

    def test_resume_with_range(self):
        data = self.hub.files["model.gguf"]
        half = len(data) // 2 + 13
        with open(self.dest + pull.PART_SUFFIX, "wb") as f:
            f.write(data[:half])
        self.run_pull(f"hf:{REPO}/model.gguf")
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), data)
        self.assertEqual([h.get("Range") for h in self.hub.cdn_requests()], [f"bytes={half}-"])

    def test_server_ignoring_range_restarts_cleanly(self):
        data = self.hub.files["model.gguf"]
        with open(self.dest + pull.PART_SUFFIX, "wb") as f:
            f.write(data[:1000])
        self.hub.ignore_range = True
        self.run_pull(f"hf:{REPO}/model.gguf")
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), data)

    def test_complete_or_oversized_part_files(self):
        data = self.hub.files["model.gguf"]
        with open(self.dest + pull.PART_SUFFIX, "wb") as f:
            f.write(data)  # complete but never renamed (interrupted at the very end)
        self.run_pull(f"hf:{REPO}/model.gguf")
        self.assertEqual(self.hub.cdn_requests(), [])
        os.remove(self.dest)
        with open(self.dest + pull.PART_SUFFIX, "wb") as f:
            f.write(data + b"junk")
        self.run_pull(f"hf:{REPO}/model.gguf")
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), data)

    def test_broken_connection_is_resumed_automatically(self):
        self.hub.break_after = 120_000  # every response breaks off after this many bytes: three attempts finish the 300 kB file
        self.run_pull(f"hf:{REPO}/model.gguf")
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), self.hub.files["model.gguf"])
        self.assertEqual([h.get("Range") for h in self.hub.cdn_requests()], [None, "bytes=120000-", "bytes=240000-"])

    def test_broken_connection_keeps_the_part_and_resumes(self):
        self.hub.break_after = 100_000
        with mock.patch.object(pull, "DOWNLOAD_ATTEMPTS", 1):
            self.expect_error(f"hf:{REPO}/model.gguf", "run the command again")
        self.assertFalse(os.path.exists(self.dest))
        part = self.dest + pull.PART_SUFFIX
        self.assertEqual(os.path.getsize(part), 100_000)
        self.hub.break_after = None
        self.run_pull(f"hf:{REPO}/model.gguf")
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), self.hub.files["model.gguf"])
        self.assertEqual(self.hub.cdn_requests()[-1].get("Range"), "bytes=100000-")

    def test_checksum_mismatch_deletes_the_part(self):
        self.hub.corrupt = True
        self.expect_error(f"hf:{REPO}/model.gguf", "checksum mismatch", "deleted")
        self.assertFalse(os.path.exists(self.dest))
        self.assertFalse(os.path.exists(self.dest + pull.PART_SUFFIX))

    def test_more_bytes_than_announced(self):
        self.hub.extra_bytes = 10
        self.expect_error(f"hf:{REPO}/model.gguf", "more than")
        self.assertFalse(os.path.exists(self.dest + pull.PART_SUFFIX))

    def test_existing_file_that_does_not_match_is_left_alone(self):
        with open(self.dest, "wb") as f:
            f.write(b"something else")
        self.expect_error(f"hf:{REPO}/model.gguf", "does not match", "delete it")
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), b"something else")

    def test_error_messages(self):
        self.expect_error(f"hf:{REPO}/missing.gguf", "no such file")
        self.expect_error(f"hf:{REPO}/model.gguf@nobranch", "not found", "HTTP 404", "Invalid rev id")
        self.expect_error("hf:other/repo/model.gguf@" + COMMIT, "access denied", pull.HF_TOKEN_ENV)
        self.hub.gated = True
        self.expect_error(f"hf:{REPO}/model.gguf", "access denied", "gated", pull.HF_TOKEN_ENV)
        with socket.socket() as s:  # a port nothing listens on
            s.bind(("127.0.0.1", 0))
            closed = s.getsockname()[1]
        with mock.patch.dict(os.environ, {pull.HF_ENDPOINT_ENV: f"http://127.0.0.1:{closed}"}):
            self.expect_error(f"hf:{REPO}/model.gguf", "cannot reach", "network")

    def test_token_is_not_forwarded_to_another_host(self):
        with mock.patch.dict(os.environ, {pull.HF_TOKEN_ENV: "secret-token"}):
            self.run_pull(f"hf:{REPO}/model.gguf")
        api = [h for m, p, h in self.hub.log if not p.startswith("/cdn/")]
        self.assertTrue(api and all(h.get("Authorization") == "Bearer secret-token" for h in api))
        cdn = self.hub.cdn_requests()
        self.assertTrue(cdn and all("Authorization" not in h for h in cdn))

    def test_snapshot_downloads_every_file_with_its_hash(self):
        snap = pull.Snapshot(REPO, COMMIT, ("sub/",))
        with redirect_stderr(io.StringIO()):
            d = pull.pull_snapshot(snap, self.dir, progress=None)
        self.assertEqual(d, os.path.join(self.dir, pull.SOURCES_SUBDIR, "owner", "model-GGUF"))
        self.assertEqual(sorted(os.listdir(d)), ["config.json", "model.gguf"])  # sub/ excluded, every page of the listing read
        with redirect_stderr(io.StringIO()):
            pull.pull_snapshot(pull.Snapshot(REPO, COMMIT), self.dir, progress=None)
        with open(os.path.join(d, "sub", "tokenizer.json"), "rb") as f:
            self.assertEqual(f.read(), self.hub.files["sub/tokenizer.json"])
        # a small (non-LFS) file is verified with its git blob hash
        self.hub.files["config.json"] = b'{"a": 2}\n'
        os.remove(os.path.join(d, "config.json"))
        self.hub.corrupt = True
        with self.assertRaises(pull.PullError) as cm, redirect_stderr(io.StringIO()):
            pull.pull_snapshot(snap, self.dir, progress=None)
        self.assertIn("git-sha1", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(d, "config.json")))

    def test_snapshot_refuses_paths_outside_the_target(self):
        self.hub.extra_tree_paths = ["../../escape.txt"]
        with self.assertRaises(pull.PullError) as cm, redirect_stderr(io.StringIO()):
            pull.pull_snapshot(pull.Snapshot(REPO, COMMIT), self.dir, progress=None)
        self.assertIn("leaves the target directory", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "escape.txt")))

    def test_conversion_needs_a_source_checkout(self):
        with mock.patch.dict(os.environ, {"DOOMS_SOURCE_DIR": self.dir}):
            for name in pull.CONVERTED:
                msg = self.expect_error(name, "no converters", "never redistributed", "pip install", f"pull {name}")
                self.assertEqual("lerobot" in msg, name == "smolvla")
        self.assertEqual(self.hub.log, [])  # nothing is downloaded before the preconditions hold

    @unittest.skipUnless(pull.source_dir(), "not a source checkout")
    def test_conversion_reports_a_missing_converter_environment(self):
        with mock.patch.dict(os.environ, {"DOOMS_CONVERT_PYTHON": os.path.join(self.dir, "no-python")}):
            self.expect_error("smolvlm", "DOOMS_CONVERT_PYTHON", "no Python")
        with mock.patch.dict(os.environ, {"DOOMS_CONVERT_PYTHON": sys.executable, "DOOMS_LLAMA_CPP_DIR": self.dir}):
            msg = self.expect_error("smolvlm")  # this interpreter lacks torch & co, or (if it has them) the missing llama.cpp is reported
            self.assertTrue(("cannot import" in msg and "DOOMS_CONVERT_PYTHON" in msg) or ("convert_hf_to_gguf.py" in msg and "DOOMS_LLAMA_CPP_DIR" in msg), msg)
        self.assertEqual(self.hub.log, [])

    def test_models_dir_and_list(self):
        self.assertEqual(pull.models_dir(), os.path.abspath(self.dir))
        self.assertEqual(pull.models_dir("~/x"), os.path.join(os.path.expanduser("~"), "x"))
        with mock.patch.dict(os.environ, {pull.MODELS_ENV: ""}):
            self.assertEqual(pull.models_dir(), os.path.abspath(os.path.expanduser(pull.DEFAULT_MODELS_DIR)))
        self.assertEqual(pull.list_models(), [])
        self.run_pull(f"hf:{REPO}/model.gguf")
        with open(os.path.join(self.dir, "other.gguf.part"), "wb") as f:
            f.write(b"12")
        rows = pull.list_models()
        self.assertEqual([(k, os.path.basename(p), n) for k, p, n in rows], [("gguf", "model.gguf", len(self.hub.files["model.gguf"])), ("partial", "other.gguf.part", 2)])


class Command(unittest.TestCase):
    def run_cli(self, *args, env=None):
        e = dict(os.environ, PYTHONPATH=PYTHON_DIR, **(env or {}))
        return subprocess.run([sys.executable, "-m", "dooms", *args], capture_output=True, text=True, env=e, timeout=120)

    def test_pull_and_list(self):
        hub = FakeHub()
        try:
            with tempfile.TemporaryDirectory() as d:
                r = self.run_cli("list", "--dir", d)
                self.assertEqual(r.returncode, 0)
                self.assertIn("no models", r.stderr)
                r = self.run_cli("pull", f"hf:{REPO}/model.gguf", "--dir", d, env={pull.HF_ENDPOINT_ENV: hub.endpoint})
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(r.stdout.strip(), os.path.join(d, "model.gguf"))
                self.assertIn("verified", r.stderr)
                r = self.run_cli("list", "--dir", d)
                self.assertIn("model.gguf", r.stdout)
                r = self.run_cli("pull", f"hf:{REPO}/missing.gguf", "--dir", d, env={pull.HF_ENDPOINT_ENV: hub.endpoint})
                self.assertEqual(r.returncode, 1)
                self.assertIn("dooms pull: ", r.stderr)
                self.assertIn("no such file", r.stderr)
                self.assertEqual(self.run_cli("pull").returncode, 2)
        finally:
            hub.close()


@unittest.skipUnless(os.environ.get("DOOMS_TEST_PULL") == "1", "set DOOMS_TEST_PULL=1 to download a real GGUF (~490 MB) from Hugging Face")
class RealDownload(unittest.TestCase):
    def test_official_gguf(self):
        import dooms  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {pull.HF_ENDPOINT_ENV: pull.DEFAULT_HF_ENDPOINT}):
            os.environ.pop(pull.HF_ENDPOINT_ENV)
            path = pull.pull("qwen2.5-0.5b", d)
            alias = pull.ALIASES["qwen2.5-0.5b"]
            self.assertEqual(path, os.path.join(d, alias.file))
            self.assertTrue(pull.verify(path, pull.file_info(alias.repo, alias.revision, alias.file)))
            with dooms.LLM(path, n_ctx=256) as llm:
                self.assertTrue(llm.generate("The capital of France is", 4).strip())


if __name__ == "__main__":
    unittest.main()
