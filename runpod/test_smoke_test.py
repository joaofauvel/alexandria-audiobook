import unittest
from pathlib import Path


class SmokeTestBootstrapTests(unittest.TestCase):
    def test_huggingface_environment_is_set_before_qwen_import(self):
        source = Path(__file__).with_name("smoke_test.py").read_text(encoding="utf-8")
        qwen_import = source.index("from qwen_tts import Qwen3TTSModel")
        hf_setup = source.index('os.environ.setdefault("HF_HOME"')
        hub_setup = source.index('os.environ.setdefault("HF_HUB_CACHE"')
        transformers_setup = source.index('os.environ.setdefault("TRANSFORMERS_CACHE"')

        self.assertLess(hf_setup, qwen_import)
        self.assertLess(hub_setup, qwen_import)
        self.assertLess(transformers_setup, qwen_import)


if __name__ == "__main__":
    unittest.main()
