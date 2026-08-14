import os


EXPECTED_ATTENTION = "flash_attention_2"
DEFAULT_HF_HOME = "/workspace/.cache/huggingface"

os.environ.setdefault("ALEXANDRIA_DATA_ROOT", "/workspace/alexandria")
os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"])
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(os.environ["ALEXANDRIA_DATA_ROOT"], "torchinductor-cache"),
)

import soundfile as sf
import torch
import triton
import flash_attn
from qwen_tts import Qwen3TTSModel


def _effective_attention_implementation(model):
    model_container = getattr(model, "model", None)
    talker = getattr(model_container, "talker", None)
    candidates = [
        getattr(model, "config", None),
        getattr(model_container, "config", None),
        getattr(talker, "config", None),
        getattr(getattr(talker, "model", None), "config", None),
    ]
    for config in candidates:
        if config is None:
            continue
        for attribute in ("_attn_implementation", "attn_implementation"):
            value = getattr(config, attribute, None)
            if value:
                return value
    return None


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not str(torch.version.cuda).startswith("12.8"):
        raise RuntimeError(f"Expected CUDA 12.8, found {torch.version.cuda}")

    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"triton={triton.__version__}")
    print(f"flash_attn={getattr(flash_attn, '__version__', 'installed')}")

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=EXPECTED_ATTENTION,
    )
    effective_attention = _effective_attention_implementation(model)
    print(f"attn_implementation={effective_attention}")
    if effective_attention != EXPECTED_ATTENTION:
        raise RuntimeError(
            "Qwen model did not report the required attention implementation: "
            f"{effective_attention!r}"
        )

    wavs, sample_rate = model.generate_custom_voice(
        text="Este es un texto breve para verificar la síntesis de voz.",
        language="Spanish",
        speaker="Ryan",
        instruct="Neutral narration.",
        non_streaming_mode=True,
        max_new_tokens=512,
    )
    if not wavs or sample_rate <= 0:
        raise RuntimeError("Qwen generation returned no audio")

    output_root = os.environ.get("ALEXANDRIA_DATA_ROOT", "/workspace/alexandria")
    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, "alexandria-smoke.wav")
    sf.write(output_path, wavs[0], sample_rate)
    duration = len(wavs[0]) / sample_rate
    print(f"generated={duration:.2f}s path={output_path}")


if __name__ == "__main__":
    main()
