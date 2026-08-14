import importlib.util
import sys
import types
import unittest
from unittest.mock import patch


# The loader tests do not exercise audio or CUDA code. Keep them runnable with
# the repository's host Python even when the full runtime dependencies are only
# available in the RunPod image.
if importlib.util.find_spec("numpy") is None:
    sys.modules["numpy"] = types.ModuleType("numpy")
if importlib.util.find_spec("soundfile") is None:
    sys.modules["soundfile"] = types.ModuleType("soundfile")
if importlib.util.find_spec("pydub") is None:
    pydub = types.ModuleType("pydub")
    pydub.AudioSegment = object
    sys.modules["pydub"] = pydub

from tts import TTSEngine


class FakeModel:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append((model_id, kwargs))
        return object()


class FlashAttentionLoaderTests(unittest.TestCase):
    def setUp(self):
        FakeModel.calls = []

    @patch.object(TTSEngine, "_resolve_local_model_path", return_value=None)
    def test_direct_load_forces_flash_attention_2(self, _cache_path):
        TTSEngine._load_model(FakeModel, "example/model", {"dtype": "bf16"})

        self.assertEqual(len(FakeModel.calls), 1)
        self.assertEqual(
            FakeModel.calls[0][1]["attn_implementation"],
            "flash_attention_2",
        )

    @patch.object(
        TTSEngine,
        "_resolve_local_model_path",
        return_value="/tmp/cached-model",
    )
    def test_cached_load_forces_flash_attention_2(self, _cache_path):
        TTSEngine._load_model(FakeModel, "example/model", {"dtype": "bf16"})

        self.assertEqual(len(FakeModel.calls), 1)
        self.assertEqual(
            FakeModel.calls[0][1]["attn_implementation"],
            "flash_attention_2",
        )


if __name__ == "__main__":
    unittest.main()
