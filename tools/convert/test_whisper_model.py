"""Unit tests for the Whisper converter: a tiny random WhisperForConditionalGeneration with the real whisper-tiny processor (vocabulary, mel filters,
generation config; skipped without models/whisper/whisper-tiny). Run from the repo root:
    .venv/bin/python -m unittest discover -s tools/convert -p 'test_*.py'
"""
import copy
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import convert_common as cc  # noqa: E402

PROCESSOR_DIR = os.path.join(HERE, "..", "..", "models", "whisper", "whisper-tiny")


@unittest.skipUnless(os.path.isfile(os.path.join(PROCESSOR_DIR, "generation_config.json")), "needs models/whisper/whisper-tiny (processor files)")
class WhisperConverterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch  # noqa: PLC0415
        from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor  # noqa: PLC0415
        import whisper_model as wm  # noqa: PLC0415
        cls.wm, cls.torch = wm, torch
        cls.processor = WhisperProcessor.from_pretrained(PROCESSOR_DIR)
        cls.gen = GenerationConfig.from_pretrained(PROCESSOR_DIR)
        cls.config = WhisperConfig(vocab_size=51865, num_mel_bins=80, d_model=64, encoder_layers=1, encoder_attention_heads=1, decoder_layers=2,
                                   decoder_attention_heads=1, encoder_ffn_dim=64, decoder_ffn_dim=96, max_source_positions=1500, max_target_positions=448)
        torch.manual_seed(0)
        cls.model = WhisperForConditionalGeneration(cls.config).eval()

    def model_with(self, **changes):
        from transformers import WhisperForConditionalGeneration  # noqa: PLC0415
        cfg = copy.deepcopy(self.config)
        for k, v in changes.items():
            setattr(cfg, k, v)
        return WhisperForConditionalGeneration(cfg).eval()

    def write(self, model):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = os.path.join(tmp.name, "decoder.gguf")
        self.wm.write_decoder(model, self.processor, self.gen, out)
        return out

    def test_written_file_has_the_layout_and_values(self):
        import gguf  # noqa: PLC0415
        r = gguf.GGUFReader(self.write(self.model))
        kv = {f.name: f for f in r.fields.values()}

        def scalar(name):
            f = kv["whisper." + name]
            return f.parts[f.data[0]][0]

        self.assertEqual(scalar("n_text_layer"), 2)
        self.assertEqual(scalar("n_text_ff"), 96)
        self.assertEqual(scalar("n_audio_ctx"), 1500)
        self.assertEqual(scalar("chunk_samples"), 480000)
        tok = self.processor.tokenizer
        self.assertEqual(scalar("token.sot"), tok.convert_tokens_to_ids("<|startoftranscript|>"))
        self.assertEqual(scalar("token.eot"), tok.convert_tokens_to_ids("<|endoftext|>"))
        self.assertEqual(scalar("token.timestamp_begin"), tok.convert_tokens_to_ids("<|0.00|>"))
        t = {x.name: x for x in r.tensors}
        self.assertEqual(sum(1 for n in t if n.startswith("blk.")), 2 * 24)
        mel = np.asarray(t["mel_filters"].data, dtype=np.float32).reshape(80, 201)
        self.assertTrue(np.allclose(mel, np.asarray(self.processor.feature_extractor.mel_filters, dtype=np.float32).T))
        fc1 = np.asarray(t["blk.1.fc1.weight"].data, dtype=np.float32).reshape(96, 64)
        self.assertTrue(np.allclose(fc1, self.model.model.decoder.layers[1].fc1.weight.detach().numpy(), atol=1e-3))
        self.assertNotIn("blk.0.attn_k.bias", t)

    def test_languages_and_special_tokens(self):
        codes, ids = self.wm.languages(self.gen)
        self.assertEqual(codes[0], "en")
        self.assertEqual(ids[0], self.processor.tokenizer.convert_tokens_to_ids("<|en|>"))
        self.assertEqual(len(codes), len(set(codes)))
        st = self.wm.special_tokens(self.processor, self.gen)
        # whisper-tiny's older vocabulary names it <|nocaptions|>; "<|nospeech|>" would map to unk (= <|endoftext|>)
        self.assertIn(self.processor.tokenizer.convert_ids_to_tokens(st["nospeech"]), ("<|nospeech|>", "<|nocaptions|>"))
        self.assertNotEqual(st["nospeech"], st["eot"])

    def test_refuses_what_the_engine_does_not_implement(self):
        cases = {
            "activation": self.model_with(activation_function="relu"),
            "scale_embedding": self.model_with(scale_embedding=True),
            "head_dim": self.model_with(decoder_attention_heads=2),          # 32-wide heads
            "ffn multiple of 32": self.model_with(decoder_ffn_dim=100),
        }
        for what, model in cases.items():
            with self.subTest(what), self.assertRaises(cc.ConversionError):
                self.wm.check_supported(model, self.processor)
        untied = self.model_with()
        untied.proj_out.weight = self.torch.nn.Parameter(untied.proj_out.weight.detach().clone() + 1.0)
        with self.assertRaises(cc.ConversionError):
            self.wm.check_supported(untied, self.processor)
        kbias = self.model_with()
        L = kbias.model.decoder.layers[1].encoder_attn
        L.k_proj = self.torch.nn.Linear(64, 64, bias=True)
        with self.assertRaises(cc.ConversionError):
            self.wm.check_supported(kbias, self.processor)
        wrong_mels = self.model_with(num_mel_bins=128)
        with self.assertRaises(cc.ConversionError):
            self.wm.check_supported(wrong_mels, self.processor)

    def test_refused_conversion_writes_nothing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = os.path.join(tmp.name, "decoder.gguf")
        with self.assertRaises(cc.ConversionError):
            self.wm.write_decoder(self.model_with(activation_function="relu"), self.processor, self.gen, out)
        self.assertEqual(os.listdir(tmp.name), [])

    def test_verify_catches_a_wrong_tensor(self):
        out = self.write(self.model)
        other = self.model_with()   # same shapes, other random weights
        with self.assertRaises(cc.ConversionError):
            self.wm.verify_written(out, other)


if __name__ == "__main__":
    unittest.main()
