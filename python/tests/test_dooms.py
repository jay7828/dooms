"""Tests of the Python binding. Error paths and argument checks always run; the real-model tests run when DOOMS_MODELS points at a models directory with smolvlm/ and smolvla/ (the git-ignored `models/` of the repo) and DOOMS_PHOTOS at a directory with test photos."""
import os
import re
import subprocess
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dooms  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MODELS = os.environ.get("DOOMS_MODELS", os.path.join(ROOT, "models"))
PHOTOS = os.environ.get("DOOMS_PHOTOS", os.path.join(ROOT, "tests", "data"))


def photo(name):
    from PIL import Image  # noqa: PLC0415
    return np.asarray(Image.open(os.path.join(PHOTOS, name)).convert("RGB"))


def have(*parts):
    return all(os.path.exists(os.path.join(MODELS, p)) for p in parts) and os.path.isdir(PHOTOS)


LLM_FILE = os.environ.get("DOOMS_LLM_FILE", "")


class Llm(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(dooms.DoomsError) as cm:
            dooms.LLM("/nonexistent/model.gguf")
        self.assertIn(cm.exception.code, ("NotFound", "Unsupported"))

    @unittest.skipUnless(LLM_FILE and os.path.exists(LLM_FILE), "set DOOMS_LLM_FILE to a chat-tuned GGUF")
    def test_generate_chat_stream_and_errors(self):
        with dooms.LLM(LLM_FILE, n_ctx=1024) as llm:
            self.assertEqual(llm.context_size, 1024)
            a = llm.generate("The capital of France is", 8)
            self.assertEqual(a, llm.generate("The capital of France is", 8))  # greedy is deterministic
            pieces = []
            b = llm.generate("The capital of France is", 8, on_text=pieces.append)
            self.assertEqual(a, b)
            self.assertEqual("".join(pieces), b)
            self.assertEqual(a, "".join(llm.stream("The capital of France is", 8)))
            msgs = [{"role": "user", "content": "Reply with the single word: pong"}]
            self.assertTrue(llm.chat(msgs, 8).strip())
            self.assertEqual(llm.chat(msgs, 8, temperature=0.9, seed=5), llm.chat(msgs, 8, temperature=0.9, seed=5))
            cut = llm.generate("1 2 3 4 5 6 7 8 9", 24, stop=["5"])
            self.assertNotIn("5", cut)
            with self.assertRaises(dooms.DoomsError):
                llm.generate("", 8)
            with self.assertRaises(dooms.DoomsError):
                llm.generate("hi", 8, temperature=-1.0)
            with self.assertRaises(ValueError):
                llm.chat([], 8)
            with self.assertRaises(RuntimeError):  # stop early from the callback raising
                llm.generate("hi", 8, on_text=lambda p: (_ for _ in ()).throw(RuntimeError("boom")))


class ErrorPaths(unittest.TestCase):
    def test_version(self):
        self.assertRegex(dooms.version(), r"^\d+\.\d+\.\d+$")

    def test_missing_files_raise_not_found(self):
        with self.assertRaises(dooms.DoomsError) as cm:
            dooms.VLM("/nonexistent/v.mlpackage", "/nonexistent/t.gguf", "/nonexistent/c.gguf")
        self.assertIn(cm.exception.code, ("NotFound", "Unsupported"))
        with self.assertRaises(dooms.DoomsError):
            dooms.VLA.from_dir("/nonexistent")


    def test_images_must_be_uint8_rgb(self):
        from dooms import _image  # noqa: PLC0415
        for bad in (np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.float32), np.zeros((4, 4, 4), np.uint8)):
            with self.assertRaises(ValueError):
                _image(bad)
        img, keep = _image(np.zeros((5, 7, 3), np.uint8)[:, ::-1])  # a non-contiguous view is copied
        self.assertEqual((img.width, img.height), (7, 5))


@unittest.skipUnless(have("smolvlm/vision-fp16.mlpackage", "smolvlm/text-f16.gguf"), "SmolVLM files not available")
class Vlm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vlm = dooms.VLM.from_dir(os.path.join(MODELS, "smolvlm"))

    @classmethod
    def tearDownClass(cls):
        cls.vlm.close()

    def test_answer_equals_the_reference_model(self):
        # transformers' greedy answer for this photo and prompt (tests/unit/test_vlm_pipeline.cpp checks the same tokens)
        answer = self.vlm.generate(photo("test1.jpg"), "Describe this image in one sentence.", 64)
        self.assertEqual(answer.strip(), "In this image, we can see a person holding a spoon and a plate.")

    def test_streaming_pieces_join_to_the_full_answer(self):
        prompt = "Describe this image in one sentence."
        pieces = []
        text = self.vlm.generate(photo("test1.jpg"), prompt, 64, on_text=pieces.append)
        self.assertGreater(len(pieces), 3)
        self.assertEqual("".join(pieces), text)
        self.assertEqual(text, self.vlm.generate(photo("test1.jpg"), prompt, 64))
        self.assertEqual("".join(self.vlm.stream(photo("test1.jpg"), prompt, 64)), text)

    def test_callback_can_stop_and_errors_propagate(self):
        prompt = "Describe this image in one sentence."
        pieces = []
        short = self.vlm.generate(photo("test1.jpg"), prompt, 64, on_text=lambda p: (pieces.append(p), len(pieces) < 3)[1])
        self.assertEqual(len(pieces), 3)
        self.assertEqual(short, "".join(pieces))

        def boom(piece):
            raise KeyError("from the callback")
        with self.assertRaises(KeyError):
            self.vlm.generate(photo("test1.jpg"), prompt, 64, on_text=boom)
        # the model is still usable afterwards, and breaking out of a stream() loop stops it cleanly
        it = self.vlm.stream(photo("test1.jpg"), prompt, 64)
        first = next(it)
        it.close()
        self.assertTrue(first)
        self.assertEqual(self.vlm.generate(photo("test1.jpg"), prompt, 64).strip(), "In this image, we can see a person holding a spoon and a plate.")

    def test_token_limit_and_arguments(self):
        short = self.vlm.generate(photo("test1.jpg"), "Describe this image in one sentence.", 3)
        self.assertTrue(0 < len(short) < 40)
        with self.assertRaises(dooms.DoomsError) as cm:
            self.vlm.generate(photo("test1.jpg"), "", 8)
        self.assertEqual(cm.exception.code, "InvalidArgument")
        with self.assertRaises(dooms.DoomsError):
            self.vlm.generate(photo("test1.jpg"), "hi", 0)

    def test_closed_model_is_rejected(self):
        v = dooms.VLM.from_dir(os.path.join(MODELS, "smolvlm"))
        v.close()
        v.close()  # idempotent
        with self.assertRaises(ValueError):
            v.generate(photo("test1.jpg"), "hi", 4)


@unittest.skipUnless(have("smolvla/prefix-fp16.mlpackage", "smolvla/reference.npz", "smolvla/vocab.gguf"), "SmolVLA files not available")
class Vla(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vla = dooms.VLA.from_dir(os.path.join(MODELS, "smolvla"), seed=7)
        cls.ref = dict(np.load(os.path.join(MODELS, "smolvla", "reference.npz")))

    @classmethod
    def tearDownClass(cls):
        cls.vla.close()

    def cameras(self):
        return [np.round(self.ref[f"image_camera{k}"] * 255).astype(np.uint8).transpose(1, 2, 0) for k in (1, 2, 3)]

    def test_info(self):
        self.assertEqual((self.vla.n_cameras, self.vla.chunk_size, self.vla.action_dim, self.vla.state_dim), (3, 50, 32, 32))

    def test_chunk_matches_lerobot_given_the_same_noise(self):
        # the reference images are floats in 0..1; the API takes 8-bit pixels, so allow the 1/255 quantization on top of the Neural Engine's fp16 error
        got = self.vla.infer(self.cameras(), "pick up the red cube and place it in the box", self.ref["state"], noise=self.ref["noise"])
        want = self.ref["actions"]
        rel = np.linalg.norm(got - want) / np.linalg.norm(want)
        self.assertEqual(got.shape, want.shape)
        self.assertLess(rel, 6e-2, f"relative L2 {rel}")

    def test_noise_is_seeded_and_reproducible(self):
        cams = self.cameras()
        task = "pick up the red cube and place it in the box"
        first = dooms.VLA.from_dir(os.path.join(MODELS, "smolvla"), seed=3)
        a = first.infer(cams, task, self.ref["state"])
        c = first.infer(cams, task, self.ref["state"])  # the next call uses seed + 1
        again = dooms.VLA.from_dir(os.path.join(MODELS, "smolvla"), seed=3)
        b = again.infer(cams, task, self.ref["state"])
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.allclose(a, c))
        self.assertTrue(np.isfinite(a).all() and np.isfinite(c).all())

    def test_argument_errors(self):
        cams = self.cameras()
        with self.assertRaises(ValueError):
            self.vla.infer(cams[:2], "x", self.ref["state"])
        with self.assertRaises(dooms.DoomsError) as cm:
            self.vla.infer(cams, "", self.ref["state"])
        self.assertEqual(cm.exception.code, "InvalidArgument")
        with self.assertRaises(dooms.DoomsError):
            self.vla.infer(cams, "x", np.zeros(self.vla.state_dim + 1))
        with self.assertRaises(ValueError):
            self.vla.infer(cams, "x", self.ref["state"], noise=np.zeros((2, 2)))




if __name__ == "__main__":
    unittest.main()
