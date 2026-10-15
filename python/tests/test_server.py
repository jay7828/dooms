"""Tests of `dooms serve` (python/dooms/server.py) and the `dooms` command.

Always run: request validation and OpenAI error shapes, routing, SSE framing, serialization of concurrent requests and disconnects, against a scripted stand-in for the
model. With DOOMS_LLM_FILE (a chat-tuned GGUF): the same server over the real model (greedy chat, streamed == non-streamed, completions, stop strings, finish_reason
"length", usage counts against the tokenizer, concurrent requests), and the official `openai` client when it is importable.
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dooms  # noqa: E402
from dooms import server as srv  # noqa: E402

LLM_FILE = os.environ.get("DOOMS_LLM_FILE", "")
PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WAIT_SECONDS = 20.0  # upper bound for a background event in these tests (a generation stopping, a thread finishing); they normally take milliseconds


class FakeLLM:
    """Stands in for dooms.LLM: answers with ANSWER (one piece per token), counts a token per word, records how many generations overlap."""
    ANSWER = ["Hello", " there", ",", " this", " is", " a", " scripted", " answer", "."]
    context_size = 64

    def __init__(self, delay=0.0):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._last = None
        self.closed = False

    def count_tokens(self, text, add_special=True):
        return len(text.split()) + (1 if add_special else 0)

    def count_chat_tokens(self, messages):
        return sum(len(m["content"].split()) + 2 for m in messages) + 1

    @property
    def last_generation(self):
        return self._last

    def _run(self, n_prompt, max_new_tokens, stop, on_text):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            text, n, finish = "", 0, "length"
            for piece in self.ANSWER:
                if n == max_new_tokens:
                    break
                n += 1
                if self.delay:
                    time.sleep(self.delay)
                if stop and any(s in text + piece for s in stop):
                    cut = min((text + piece).find(s) for s in stop if s in text + piece)
                    piece = (text + piece)[len(text):cut]
                    finish = "stop"
                if piece and on_text is not None and on_text(piece) is False:
                    finish = "cancelled"
                    text += piece
                    break
                text += piece
                if finish == "stop":
                    break
            else:
                finish = "stop"
            self._last = dooms.Generation(n_prompt, n, finish)
            return text
        finally:
            self.active -= 1

    def chat(self, messages, max_new_tokens, *, stop=None, on_text=None, **_):
        return self._run(self.count_chat_tokens(messages), max_new_tokens, stop, on_text)

    def generate(self, prompt, max_new_tokens, *, stop=None, on_text=None, **_):
        return self._run(self.count_tokens(prompt), max_new_tokens, stop, on_text)


def start_server(llm, name="test-model"):
    httpd = srv.make_server(llm, name, "127.0.0.1", 0, quiet=True)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t


def request(port, method, path, body=None, raw=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    conn.request(method, path, body=data, headers=dict({"Content-Type": "application/json"}, **(headers or {})))
    r = conn.getresponse()
    payload = r.read()
    conn.close()
    return r.status, r.getheader("Content-Type"), payload


def post(port, path, body):
    status, ctype, payload = request(port, "POST", path, body)
    return status, (json.loads(payload) if ctype == "application/json" else payload.decode())


def recv_until(sock, marker: bytes) -> bytes:
    """Reads from a raw socket until `marker` arrived or the peer closed (headers and body may come in separate packets)."""
    data = b""
    while marker not in data:
        more = sock.recv(65536)
        if not more:
            break
        data += more
    return data


def sse_events(text):
    """Splits an SSE body into its data payloads, checking the framing (each event is exactly `data: ...` followed by a blank line)."""
    assert text.endswith("\n\n"), text[-50:]
    events = []
    for block in text[:-2].split("\n\n"):
        assert block.startswith("data: ") and "\n" not in block, block
        events.append(block[len("data: "):])
    return events


class ErrorShape:
    def assertOpenAIError(self, status, body, want_status, code=None, param=None):  # noqa: N802 - unittest naming
        self.assertEqual(status, want_status, body)
        self.assertIsInstance(body, dict)
        self.assertEqual(set(body), {"error"})
        self.assertEqual(set(body["error"]), {"message", "type", "param", "code"})
        self.assertTrue(body["error"]["message"])
        if code is not None:
            self.assertEqual(body["error"]["code"], code, body)
        if param is not None:
            self.assertEqual(body["error"]["param"], param, body)


class Validation(unittest.TestCase, ErrorShape):
    """parse_*_request: what is accepted, normalized and rejected (no server, no model)."""

    def chat(self, **kw):
        return srv.parse_chat_request(dict({"model": "m", "messages": [{"role": "user", "content": "hi"}]}, **kw))

    def rejects(self, fn, body, code, param=None):
        with self.assertRaises(srv.ApiError) as cm:
            fn(body)
        e = cm.exception
        self.assertOpenAIError(e.status, e.body(), 400 if code != "model_not_found" else 404, code, param)

    def test_chat_defaults_follow_openai(self):
        r = self.chat()
        self.assertEqual(r["sampling"]["temperature"], srv.DEFAULT_TEMPERATURE)
        self.assertEqual(r["sampling"]["top_p"], srv.DEFAULT_TOP_P)
        self.assertEqual(r["sampling"]["top_k"], 0)
        self.assertIsNone(r["max_tokens"])
        self.assertFalse(r["stream"])
        self.assertEqual(r["stop"], [])
        self.assertTrue(0 <= r["sampling"]["seed"] < srv.SEED_MODULUS)

    def test_chat_parameters_are_mapped(self):
        r = self.chat(max_completion_tokens=7, temperature=0.5, top_p=0.9, seed=2 ** 40 + 3, stop="x", stream=True, stream_options={"include_usage": True},
                      frequency_penalty=0, presence_penalty=0.0, n=1, user="u", logprobs=False, tools=[], response_format={"type": "text"})
        self.assertEqual(r["max_tokens"], 7)
        self.assertEqual(r["sampling"]["temperature"], 0.5)
        self.assertEqual(r["sampling"]["top_p"], 0.9)
        self.assertEqual(r["sampling"]["seed"], 3)  # reduced modulo 2^32
        self.assertEqual(r["stop"], ["x"])
        self.assertTrue(r["stream"] and r["include_usage"])
        self.assertEqual(self.chat(max_tokens=5)["max_tokens"], 5)
        self.assertEqual(self.chat(max_tokens=5, max_completion_tokens=5)["max_tokens"], 5)
        self.assertEqual(self.chat(stop=["a", "b"])["stop"], ["a", "b"])

    def test_greedy_requests(self):
        self.assertEqual(self.chat(temperature=0)["sampling"]["temperature"], 0.0)
        g = self.chat(top_p=0, temperature=0.7)["sampling"]  # top_p 0 keeps only the best token
        self.assertEqual((g["temperature"], g["top_p"]), (0.0, 1.0))

    def test_messages_are_normalized(self):
        r = srv.parse_chat_request({"messages": [{"role": "developer", "content": "be brief"},
                                                 {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "name": "x"},
                                                 {"role": "assistant", "content": "ok"}]})
        self.assertEqual(r["messages"], [{"role": "system", "content": "be brief"}, {"role": "user", "content": "a" + srv.TEXT_PART_SEPARATOR + "b"},
                                         {"role": "assistant", "content": "ok"}])
        self.assertIsNone(r["model"])

    def test_unsupported_features_are_400(self):
        c = srv.parse_chat_request
        base = {"messages": [{"role": "user", "content": "hi"}]}
        for extra, param in [({"n": 2}, "n"), ({"tools": [{"type": "function", "function": {"name": "f"}}]}, "tools"), ({"logprobs": True}, "logprobs"),
                             ({"top_logprobs": 2}, "top_logprobs"), ({"frequency_penalty": 0.5}, "frequency_penalty"), ({"presence_penalty": -1}, "presence_penalty"),
                             ({"logit_bias": {"1": 5}}, "logit_bias"), ({"response_format": {"type": "json_object"}}, "response_format"),
                             ({"tool_choice": "auto"}, "tool_choice"), ({"modalities": ["text", "audio"]}, "modalities"), ({"audio": {"voice": "x"}}, "audio")]:
            with self.subTest(param=param):
                self.rejects(c, dict(base, **extra), "unsupported_parameter", param)
        self.rejects(c, {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://x"}}]}]}, "unsupported_parameter",
                     "messages[0].content[0].type")
        self.rejects(c, {"messages": [{"role": "tool", "content": "x", "tool_call_id": "1"}]}, "unsupported_parameter", "messages[0].role")
        self.rejects(c, {"messages": [{"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}]}, "unsupported_parameter", "messages[0].tool_calls")
        p = srv.parse_completion_request
        for extra, param in [({"echo": True}, "echo"), ({"best_of": 3}, "best_of"), ({"logprobs": 1}, "logprobs"), ({"suffix": "x"}, "suffix"),
                             ({"prompt": ["a", "b"]}, "prompt"), ({"prompt": [1, 2, 3]}, "prompt")]:
            with self.subTest(param=param):
                self.rejects(p, dict({"prompt": "hi"}, **extra), "unsupported_parameter", param)

    def test_invalid_values_are_400(self):
        c = srv.parse_chat_request
        ok = [{"role": "user", "content": "hi"}]
        for body, code, param in [
            ([], "invalid_type", None), ({"messages": []}, "invalid_value", "messages"), ({"messages": "hi"}, "invalid_type", "messages"),
            ({"messages": ["hi"]}, "invalid_type", "messages[0]"), ({"messages": [{"role": "robot", "content": "x"}]}, "invalid_value", "messages[0].role"),
            ({"messages": [{"role": "user"}]}, "invalid_type", "messages[0].content"), ({"messages": ok, "temperature": 2.5}, "invalid_value", "temperature"),
            ({"messages": ok, "temperature": "hot"}, "invalid_value", "temperature"), ({"messages": ok, "top_p": 1.5}, "invalid_value", "top_p"),
            ({"messages": ok, "max_tokens": 0}, "invalid_value", "max_tokens"), ({"messages": ok, "max_tokens": 2.5}, "invalid_value", "max_tokens"),
            ({"messages": ok, "max_tokens": True}, "invalid_value", "max_tokens"), ({"messages": ok, "max_tokens": 4, "max_completion_tokens": 5}, "invalid_value", "max_tokens"),
            ({"messages": ok, "seed": 1.5}, "invalid_value", "seed"), ({"messages": ok, "stop": ["a"] * 5}, "invalid_value", "stop"),
            ({"messages": ok, "stop": [""]}, "invalid_value", "stop"), ({"messages": ok, "stop": 3}, "invalid_type", "stop"),
            ({"messages": ok, "stream": "yes"}, "invalid_type", "stream"), ({"messages": ok, "stream_options": {"include_usage": True}}, "invalid_value", "stream_options"),
            ({"messages": ok, "stream": True, "stream_options": {"x": 1}}, "unknown_parameter", "stream_options.x"),
            ({"messages": ok, "model": 3}, "invalid_type", "model"), ({"messages": ok, "temprature": 1}, "unknown_parameter", "temprature"),
        ]:
            with self.subTest(body=body):
                self.rejects(c, body, code, param)
        self.rejects(srv.parse_completion_request, {"prompt": ""}, "invalid_value", "prompt")
        self.rejects(srv.parse_completion_request, {"prompt": None}, "invalid_type", "prompt")
        self.rejects(srv.parse_completion_request, {"prompt": "x", "messages": []}, "unknown_parameter", "messages")
        self.assertEqual(srv.parse_completion_request({"prompt": ["only one"]})["prompt"], "only one")


class HttpWithScriptedModel(unittest.TestCase, ErrorShape):
    """The HTTP layer over FakeLLM: routes, framing, errors, usage, serialization and disconnects."""

    def setUp(self):
        self.llm = FakeLLM()
        self.httpd, self.thread = start_server(self.llm)
        self.httpd.service.last_generation = None
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(WAIT_SECONDS)

    def test_health_and_models(self):
        status, _, body = request(self.port, "GET", "/health")
        self.assertEqual((status, json.loads(body)["status"]), (200, "ok"))
        status, _, body = request(self.port, "GET", "/v1/models")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(data["object"], "list")
        self.assertEqual([m["id"] for m in data["data"]], ["test-model"])
        self.assertEqual(set(data["data"][0]), {"id", "object", "created", "owned_by"})
        status, _, body = request(self.port, "GET", "/v1/models/test-model")
        self.assertEqual((status, json.loads(body)["id"]), (200, "test-model"))
        status, _, body = request(self.port, "GET", "/v1/models/other")
        self.assertOpenAIError(status, json.loads(body), 404, "model_not_found", "model")

    def test_routing_errors(self):
        status, _, body = request(self.port, "GET", "/v2/nothing")
        self.assertOpenAIError(status, json.loads(body), 404, "unknown_url")
        status, body = post(self.port, "/v1/embeddings", {"input": "x"})
        self.assertOpenAIError(status, body, 404, "unknown_url")
        status, _, body = request(self.port, "GET", "/v1/chat/completions")
        self.assertOpenAIError(status, json.loads(body), 405, "method_not_allowed")

    def test_malformed_bodies(self):
        for raw in (b"{not json", b"", b"\xff\xfe", b"[1, 2]", b"null"):
            with self.subTest(raw=raw):
                status, _, body = request(self.port, "POST", "/v1/chat/completions", raw=raw)
                self.assertOpenAIError(status, json.loads(body), 400)
        status, body = post(self.port, "/v1/chat/completions", {"model": "other", "messages": [{"role": "user", "content": "hi"}]})
        self.assertOpenAIError(status, body, 404, "model_not_found", "model")

    def test_oversized_body_is_413_without_reading_it(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=WAIT_SECONDS) as s:
            s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % (srv.MAX_BODY_BYTES + 1))
            reply = recv_until(s, b"request_too_large").decode()
        self.assertTrue(reply.startswith("HTTP/1.1 413"), reply[:80])
        self.assertIn('"request_too_large"', reply)

    def test_chat_response_shape_and_usage(self):
        msgs = [{"role": "user", "content": "say something nice"}]
        status, body = post(self.port, "/v1/chat/completions", {"model": "test-model", "messages": msgs})
        self.assertEqual(status, 200, body)
        self.assertTrue(body["id"].startswith("chatcmpl-"))
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "test-model")
        self.assertIsInstance(body["created"], int)
        self.assertEqual(len(body["choices"]), 1)
        ch = body["choices"][0]
        self.assertEqual(ch["index"], 0)
        self.assertEqual(ch["message"]["role"], "assistant")
        self.assertEqual(ch["message"]["content"], "".join(FakeLLM.ANSWER))
        self.assertEqual(ch["finish_reason"], "stop")
        n_prompt = self.llm.count_chat_tokens(msgs)
        self.assertEqual(body["usage"], {"prompt_tokens": n_prompt, "completion_tokens": len(FakeLLM.ANSWER), "total_tokens": n_prompt + len(FakeLLM.ANSWER)})

    def test_completion_shape_length_and_stop(self):
        status, body = post(self.port, "/v1/completions", {"prompt": "one two", "max_tokens": 3})
        self.assertEqual(status, 200, body)
        self.assertTrue(body["id"].startswith("cmpl-"))
        self.assertEqual(body["object"], "text_completion")
        self.assertEqual(body["choices"][0]["text"], "".join(FakeLLM.ANSWER[:3]))
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(body["usage"]["completion_tokens"], 3)
        status, body = post(self.port, "/v1/completions", {"prompt": "one two", "stop": [" is"]})
        self.assertEqual(body["choices"][0]["text"], "Hello there, this")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

    def test_context_limit(self):
        status, body = post(self.port, "/v1/completions", {"prompt": "word " * 10, "max_tokens": FakeLLM.context_size})
        self.assertOpenAIError(status, body, 400, "context_length_exceeded", "prompt")
        status, body = post(self.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "w " * FakeLLM.context_size}]})
        self.assertOpenAIError(status, body, 400, "context_length_exceeded", "messages")
        # without max_tokens the answer may use the rest of the context (OpenAI's behaviour)
        status, body = post(self.port, "/v1/completions", {"prompt": "word " * (FakeLLM.context_size - 3)})
        self.assertEqual((status, body["choices"][0]["finish_reason"]), (200, "length"))
        self.assertEqual(body["usage"]["completion_tokens"], 2)

    def test_streamed_chat_framing(self):
        status, ctype, raw = request(self.port, "POST", "/v1/chat/completions",
                                     {"messages": [{"role": "user", "content": "hi"}], "stream": True, "stream_options": {"include_usage": True}})
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "text/event-stream")
        events = sse_events(raw.decode())
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        ids = {c["id"] for c in chunks}
        self.assertEqual(len(ids), 1)
        self.assertTrue(all(c["object"] == "chat.completion.chunk" and c["model"] == "test-model" for c in chunks))
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": ""})
        content = [c["choices"][0]["delta"]["content"] for c in chunks[1:-2]]
        self.assertEqual(content, FakeLLM.ANSWER)
        self.assertEqual(chunks[-2]["choices"][0], {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"})
        self.assertTrue(all(c["usage"] is None for c in chunks[:-1]))
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], len(FakeLLM.ANSWER))

    def test_streamed_completion_framing(self):
        status, _, raw = request(self.port, "POST", "/v1/completions", {"prompt": "x", "stream": True, "max_tokens": 2})
        events = sse_events(raw.decode())
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertTrue(all(c["object"] == "text_completion" and "usage" not in c for c in chunks))
        self.assertEqual([c["choices"][0]["text"] for c in chunks], FakeLLM.ANSWER[:2] + [""])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")

    def test_errors_before_a_stream_are_plain_json(self):
        status, body = post(self.port, "/v1/completions", {"prompt": "word " * 100, "stream": True})
        self.assertOpenAIError(status, body, 400, "context_length_exceeded")

    def test_keep_alive_serves_several_requests_on_one_connection(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=WAIT_SECONDS)
        for body in ({"prompt": "a", "max_tokens": 1}, {"prompt": "a", "max_tokens": 1, "stream": True}, {"prompt": "a", "max_tokens": 2}):
            conn.request("POST", "/v1/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            self.assertEqual(r.status, 200)
            r.read()
        conn.close()

    def test_concurrent_requests_are_serialized(self):
        self.llm.delay = 0.01
        results, errors = {}, []

        def worker(i):
            try:
                results[i] = post(self.port, "/v1/completions", {"prompt": f"request {i}", "max_tokens": 4})
            except Exception as e:  # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(WAIT_SECONDS)
        self.assertEqual(errors, [])
        self.assertEqual(self.llm.max_active, 1)
        for i, (status, body) in results.items():
            self.assertEqual(status, 200)
            self.assertEqual(body["choices"][0]["text"], "".join(FakeLLM.ANSWER[:4]))
            self.assertEqual(body["usage"]["prompt_tokens"], self.llm.count_tokens(f"request {i}"))
        self.assertEqual(len(results), 6)

    def test_client_disconnect_stops_generation(self):
        self.llm.delay = 0.05
        FakeLLM.ANSWER = FakeLLM.ANSWER * 6  # long enough that the client leaves early
        try:
            s = socket.create_connection(("127.0.0.1", self.port), timeout=WAIT_SECONDS)
            body = json.dumps({"prompt": "x", "stream": True, "max_tokens": 50}).encode()
            s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
            self.assertIn(b"data: ", recv_until(s, b"data: "))
            s.shutdown(socket.SHUT_RDWR)
            s.close()
            deadline = time.time() + WAIT_SECONDS
            while self.httpd.service.last_generation is None and time.time() < deadline:
                time.sleep(0.01)
            g = self.httpd.service.last_generation
            self.assertIsNotNone(g)
            self.assertEqual(g.finish_reason, "cancelled")
            self.assertLess(g.completion_tokens, 50)
            # the lock was released: the next request is served
            status, body = post(self.port, "/v1/completions", {"prompt": "x", "max_tokens": 1})
            self.assertEqual(status, 200)
        finally:
            FakeLLM.ANSWER = FakeLLM.ANSWER[:9]


class Command(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, PYTHONPATH=PYTHON_DIR)
        return subprocess.run([sys.executable, "-m", "dooms", *args], capture_output=True, text=True, env=env, timeout=120)

    def test_version_and_usage(self):
        r = self.run_cli("--version")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(dooms.version(), r.stdout)
        self.assertEqual(self.run_cli().returncode, 2)
        self.assertEqual(self.run_cli("serve").returncode, 2)  # --model is required

    def test_serve_errors(self):
        r = self.run_cli("serve", "--model", "/nonexistent/model.gguf")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no such model file", r.stderr)
        r = self.run_cli("serve", "--model", "/nonexistent/model.gguf", "--port", "70000")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--port", r.stderr)

    @unittest.skipUnless(LLM_FILE and os.path.exists(LLM_FILE), "set DOOMS_LLM_FILE to a chat-tuned GGUF")
    def test_serve_runs_and_answers(self):
        env = dict(os.environ, PYTHONPATH=PYTHON_DIR)
        p = subprocess.Popen([sys.executable, "-m", "dooms", "serve", "--model", LLM_FILE, "--port", "0", "--n-ctx", "512", "--model-name", "cli-test", "--quiet"],
                             stderr=subprocess.PIPE, text=True, env=env)
        try:
            line = ""
            while "http://" not in line:
                line = p.stderr.readline()
                self.assertTrue(line or p.poll() is None, "the server exited before listening")
            port = int(line.split("http://", 1)[1].split("/", 1)[0].rsplit(":", 1)[1])
            status, body = post(port, "/v1/completions", {"model": "cli-test", "prompt": "The capital of France is", "max_tokens": 4, "temperature": 0})
            self.assertEqual(status, 200, body)
            self.assertEqual(body["usage"]["completion_tokens"], 4)
            p.terminate()  # SIGTERM: a clean stop (model closed), exit code 0
            self.assertEqual(p.wait(WAIT_SECONDS), 0)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait(WAIT_SECONDS)
            p.stderr.close()


@unittest.skipUnless(LLM_FILE and os.path.exists(LLM_FILE), "set DOOMS_LLM_FILE to a chat-tuned GGUF")
class RealModel(unittest.TestCase):
    N_CTX = 1024

    @classmethod
    def setUpClass(cls):
        cls.llm = dooms.LLM(LLM_FILE, n_ctx=cls.N_CTX)
        cls.httpd, cls.thread = start_server(cls.llm, "qwen-test")
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(WAIT_SECONDS)
        cls.llm.close()  # before interpreter exit: ggml's Metal device must not outlive its buffers

    def count(self, fn, *args):
        with self.httpd.service.lock:
            return fn(*args)

    MSGS = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Name three primary colours."}]

    def test_chat_stream_equals_non_stream_for_greedy(self):
        req = {"model": "qwen-test", "messages": self.MSGS, "temperature": 0, "max_tokens": 48}
        status, body = post(self.port, "/v1/chat/completions", req)
        self.assertEqual(status, 200, body)
        text = body["choices"][0]["message"]["content"]
        self.assertTrue(text.strip())
        u = body["usage"]
        self.assertEqual(u["prompt_tokens"], self.count(self.llm.count_chat_tokens, self.MSGS))
        self.assertEqual(u["total_tokens"], u["prompt_tokens"] + u["completion_tokens"])
        if body["choices"][0]["finish_reason"] == "stop":
            self.assertLess(u["completion_tokens"], 48)
        else:
            self.assertEqual(u["completion_tokens"], 48)
        _, _, raw = request(self.port, "POST", "/v1/chat/completions", dict(req, stream=True, stream_options={"include_usage": True}))
        events = sse_events(raw.decode())
        chunks = [json.loads(e) for e in events[:-1]]
        streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        self.assertEqual(streamed, text)
        self.assertEqual(chunks[-1]["usage"], u)
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], body["choices"][0]["finish_reason"])

    def test_completion_length_and_usage_match_the_tokenizer(self):
        prompt = "Count from one to twenty: one, two, three,"
        status, body = post(self.port, "/v1/completions", {"prompt": prompt, "max_tokens": 6, "temperature": 0})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(body["usage"]["completion_tokens"], 6)
        self.assertEqual(body["usage"]["prompt_tokens"], self.count(self.llm.count_tokens, prompt))
        # the generated text re-tokenizes to the tokens generated (true for this plain-ASCII continuation; in general a re-tokenization may merge pieces)
        self.assertEqual(self.count(self.llm.count_tokens, body["choices"][0]["text"], False), 6)
        # a natural stop is "stop", below the limit
        status, body = post(self.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Reply with the single word: pong"}], "temperature": 0,
                                                                 "max_tokens": 64})
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertLess(body["usage"]["completion_tokens"], 64)

    def test_stop_strings(self):
        prompt = "Count from one to twenty: one, two, three,"
        _, full = post(self.port, "/v1/completions", {"prompt": prompt, "max_tokens": 24, "temperature": 0})
        text = full["choices"][0]["text"]
        self.assertIn(" six", text)
        _, cut = post(self.port, "/v1/completions", {"prompt": prompt, "max_tokens": 24, "temperature": 0, "stop": [" six", "zzz"]})
        self.assertEqual(cut["choices"][0]["text"], text[: text.index(" six")])
        self.assertEqual(cut["choices"][0]["finish_reason"], "stop")
        self.assertLess(cut["usage"]["completion_tokens"], 24)
        _, _, raw = request(self.port, "POST", "/v1/completions", {"prompt": prompt, "max_tokens": 24, "temperature": 0, "stop": " six", "stream": True})
        chunks = [json.loads(e) for e in sse_events(raw.decode())[:-1]]
        self.assertEqual("".join(c["choices"][0]["text"] for c in chunks), text[: text.index(" six")])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_seeded_sampling_is_reproducible(self):
        req = {"prompt": "Once upon a time", "max_tokens": 16, "temperature": 0.9, "seed": 11}
        a = post(self.port, "/v1/completions", req)[1]["choices"][0]["text"]
        b = post(self.port, "/v1/completions", req)[1]["choices"][0]["text"]
        self.assertEqual(a, b)

    def test_concurrent_requests_get_their_own_answers(self):
        prompts = ["The capital of France is", "Water is made of", "The largest planet is", "One plus one equals"]
        want = {p: post(self.port, "/v1/completions", {"prompt": p, "max_tokens": 8, "temperature": 0})[1]["choices"][0]["text"] for p in prompts}
        got, errors = {}, []

        def worker(p, stream):
            try:
                if stream:
                    _, _, raw = request(self.port, "POST", "/v1/completions", {"prompt": p, "max_tokens": 8, "temperature": 0, "stream": True})
                    got[(p, stream)] = "".join(json.loads(e)["choices"][0]["text"] for e in sse_events(raw.decode())[:-1])
                else:
                    got[(p, stream)] = post(self.port, "/v1/completions", {"prompt": p, "max_tokens": 8, "temperature": 0})[1]["choices"][0]["text"]
            except Exception as e:  # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(p, s)) for p in prompts for s in (False, True)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(WAIT_SECONDS * 3)
        self.assertEqual(errors, [])
        self.assertEqual(len(got), 2 * len(prompts))
        for (p, _), text in got.items():
            self.assertEqual(text, want[p])

    def test_disconnect_cancels_a_real_generation(self):
        body = json.dumps({"prompt": "Write a very long story about a dragon.", "stream": True, "max_tokens": 400, "temperature": 0}).encode()
        s = socket.create_connection(("127.0.0.1", self.port), timeout=WAIT_SECONDS)
        s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
        self.assertIn(b"data: ", recv_until(s, b"data: "))
        s.close()
        deadline = time.time() + WAIT_SECONDS
        g = None
        while time.time() < deadline:
            g = self.httpd.service.last_generation
            if g is not None and g.finish_reason == "cancelled":
                break
            time.sleep(0.01)
        self.assertEqual(g.finish_reason, "cancelled")
        self.assertLess(g.completion_tokens, 400)
        # the lock was released: the next request is answered
        status, _ = post(self.port, "/v1/completions", {"prompt": "x", "max_tokens": 1, "temperature": 0})
        self.assertEqual(status, 200)

    @unittest.skipUnless(__import__("importlib").util.find_spec("openai"), "the openai package is not installed")
    def test_official_openai_client(self):
        import openai  # noqa: PLC0415

        client = openai.OpenAI(base_url=f"http://127.0.0.1:{self.port}/v1", api_key="unused")
        self.assertEqual([m.id for m in client.models.list().data], ["qwen-test"])
        r = client.chat.completions.create(model="qwen-test", messages=self.MSGS, temperature=0, max_tokens=32)
        text = r.choices[0].message.content
        self.assertTrue(text.strip())
        self.assertIn(r.choices[0].finish_reason, ("stop", "length"))
        self.assertEqual(r.usage.total_tokens, r.usage.prompt_tokens + r.usage.completion_tokens)
        stream = client.chat.completions.create(model="qwen-test", messages=self.MSGS, temperature=0, max_tokens=32, stream=True,
                                                stream_options={"include_usage": True})
        parts, usage = [], None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
            if chunk.usage:
                usage = chunk.usage
        self.assertEqual("".join(parts), text)
        self.assertEqual(usage.completion_tokens, r.usage.completion_tokens)
        c = client.completions.create(model="qwen-test", prompt="The capital of France is", max_tokens=4, temperature=0)
        self.assertEqual(c.choices[0].finish_reason, "length")
        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(model="qwen-test", messages=self.MSGS, n=2)
        self.assertEqual(cm.exception.code, "unsupported_parameter")
        with self.assertRaises(openai.NotFoundError):
            client.chat.completions.create(model="gpt-4o", messages=self.MSGS)


if __name__ == "__main__":
    unittest.main()
