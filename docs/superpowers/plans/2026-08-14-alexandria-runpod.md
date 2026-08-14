# Alexandria RunPod Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a RunPod-specific Alexandria image with enforced FlashAttention 2, persistent storage, verified SSH/SCP access, and a working Spanish narration UI flow without changing Pinokio.

**Architecture:** Keep the upstream/local Pinokio path untouched. Add a RunPod Dockerfile whose dependency layers are cached separately from application source, make `app/tts.py` force Qwen's `flash_attention_2` implementation, and use a startup entrypoint to persist Alexandria's existing runtime paths under `/workspace`. GitHub Actions publishes immutable GHCR images; a manually created RunPod Pod exposes the UI on HTTP 4200 and SSH on TCP 22.

**Tech Stack:** RunPod's pinned `runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404` base (CUDA 12.8, PyTorch 2.8, Triton, `nvcc`, and `uv`), `qwen-tts==0.1.1`, `flash-attn==2.8.3`, Docker BuildKit, GitHub Actions, GHCR, RunPod Pod, FastAPI/vanilla Alexandria UI, `runpodctl`, SSH/SCP.

**Spec:** `docs/superpowers/specs/2026-08-14-alexandria-runpod-design.md`

## Global Constraints

- Keep local Pinokio scripts and the existing local Docker path unchanged.
- The RunPod image starts from the pinned official `runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404` image, which supplies CUDA 12.8, PyTorch 2.8, Triton, `nvcc`, and `uv`; do not install a second CUDA/PyTorch stack.
- Install `qwen-tts==0.1.1` and `flash-attn==2.8.3` with `uv pip --system --break-system-packages`; compile FlashAttention with `--no-build-isolation` because the official Ubuntu base marks its system interpreter as externally managed.
- Build with `MAX_JOBS=2` and `TORCH_CUDA_ARCH_LIST="8.6;8.9"`.
- Every local Qwen model load must use `attn_implementation="flash_attention_2"`; no silent eager/SDPA fallback is allowed.
- The RunPod Pod exposes `4200/http` and `22/tcp`, has a public IP for full SSH, and receives the exact registered public key as `PUBLIC_KEY`.
- SSH uses public-key authentication only; the private key never enters the image, repository, or Pod.
- Persist Alexandria data, Hugging Face cache, and TorchInductor cache under `/workspace`.
- Keep `.env` minimal; TTS settings belong in persisted Alexandria `config.json`.
- Use Alexandria's existing single-speaker Spanish workflow for the POC; do not add translation, Serverless, or remote Gradio support.
- Run GitHub Actions only on manual dispatch and version tags; use BuildKit/GHA cache.

---

## File Map

- Modify: `app/tts.py` — force FlashAttention 2 in the shared local Qwen loader.
- Create: `app/test_tts.py` — standard-library unit coverage for the loader's attention kwarg injection, with no new test dependency.
- Create: `Dockerfile.runpod` — cached RunPod image layers, CUDA build toolchain, Python dependencies, SSH server, Alexandria source, and ports.
- Create: `runpod/entrypoint.sh` — persistent-path mapping, SSH key installation, hardened sshd startup, and Alexandria process startup.
- Create: `runpod/config.example.json` — Spanish/local/single-speaker-friendly persisted configuration seed with no translation model.
- Create: `runpod/smoke_test.py` — CUDA/FlashAttention/Triton checks and one real Spanish CustomVoice generation.
- Create: `runpod/.env.example` — only the persistent data-root setting; public key is supplied at Pod creation, never committed.
- Create: `runpod/README.md` — GHCR, Pod, volume, port, SSH-key, SCP, UI, and smoke-test instructions.
- Create: `.github/workflows/runpod-image.yml` — manually/tag-triggered cached GHCR build.
- Modify: `.gitignore` — allow the committed `runpod/.env.example` despite the repository's global dotfile ignore.
- Do not modify: `install.js`, `start.js`, `torch.js`, `pinokio.js`, `pinokio.json`, existing `Dockerfile`, existing `docker-compose.yml`.

---

### Task 1: Force FlashAttention 2 in the Qwen Loader

**Files:**
- Create: `app/test_tts.py`
- Modify: `app/tts.py:463-485`

**Interfaces:**
- Consumes: existing `TTSEngine._load_model(model_cls, model_id, load_kwargs)` calls from CustomVoice, Base, VoiceDesign, and LoRA initialization.
- Produces: `_load_model` calls that always pass `attn_implementation="flash_attention_2"` to both cached-path and Hub-path `from_pretrained` calls.

- [ ] **Step 1: Write the failing unit test**

Create `app/test_tts.py` using `unittest` and `unittest.mock`; do not add a test dependency. The fake model records both `from_pretrained` calls and the test patches cache resolution to force the direct-download branch:

```python
import unittest
from unittest.mock import patch

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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd /home/julia/devel/alexandria-audiobook/app
python -m unittest test_tts -v
```

Expected: both tests fail because the current loader forwards only `dtype` and does not add `attn_implementation`.

- [ ] **Step 3: Implement the minimal loader change**

At the start of `TTSEngine._load_model`, copy the caller's dictionary and overwrite the attention implementation before either cache branch:

```python
load_kwargs = dict(load_kwargs)
load_kwargs["attn_implementation"] = "flash_attention_2"
```

Do not add a top-level `torch`, `qwen_tts`, or `flash_attn` import. This preserves Alexandria's current optional/lazy import behavior. Do not add environment-based attention selection; FlashAttention is a RunPod image requirement, not a user-tunable setting.

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```bash
cd /home/julia/devel/alexandria-audiobook/app
python -m unittest test_tts -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Run the Python syntax check**

Start no new service for this step; verify the existing test script remains unchanged and the focused test module is importable:

```bash
cd /home/julia/devel/alexandria-audiobook/app
python -m py_compile tts.py test_tts.py
```

Expected: exit code 0.

- [ ] **Step 6: Commit the focused source change**

```bash
cd /home/julia/devel/alexandria-audiobook
git add app/tts.py app/test_tts.py
git commit -m "feat: require FlashAttention for local Qwen loads"
```

---

### Task 2: Build the RunPod Image and Persistent Entrypoint

**Files:**
- Create: `Dockerfile.runpod`
- Create: `runpod/entrypoint.sh`
- Create: `runpod/config.example.json`
- Create: `runpod/smoke_test.py`

**Interfaces:**
- Consumes: the Task 1 `TTSEngine._load_model` behavior, existing `app/requirements.txt`, `default_prompts.txt`, `review_prompts.txt`, `persona_prompts.txt`, and `builtin_lora/manifest.json`.
- Produces: an image whose default command is `python app/app.py`, whose internal ports are 4200 and 22, and whose persistent data root is controlled by `ALEXANDRIA_DATA_ROOT`.

- [ ] **Step 1: Add the RunPod Dockerfile with cache-friendly layer order**

Create `Dockerfile.runpod` with these properties:

```dockerfile
# syntax=docker/dockerfile:1.7
FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404@sha256:f46469f35597269c3e2a13866b86a12b0bd910b71008956da2c95440c59827e1

WORKDIR /alexandria

ENV DEBIAN_FRONTEND=noninteractive \
    MAX_JOBS=2 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9" \
    ALEXANDRIA_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libsndfile1 \
      openssh-server \
      sox && \
    rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt /tmp/alexandria-requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages ninja packaging psutil && \
    uv pip install --system --break-system-packages -r /tmp/alexandria-requirements.txt && \
    uv pip install --system --break-system-packages qwen-tts==0.1.1 && \
    uv pip install --system --break-system-packages flash-attn==2.8.3 --no-build-isolation

COPY app/ /alexandria/app/
COPY default_prompts.txt review_prompts.txt persona_prompts.txt /alexandria/
COPY builtin_lora/ /alexandria/builtin_lora/
COPY runpod/ /alexandria/runpod/

RUN chmod 0755 /alexandria/runpod/entrypoint.sh && \
    mkdir -p /alexandria/scripts \
      /alexandria/designed_voices \
      /alexandria/clone_voices \
      /alexandria/lora_models \
      /alexandria/lora_datasets \
      /alexandria/dataset_builder \
      /alexandria/app/uploads

EXPOSE 4200 22
ENTRYPOINT ["/alexandria/runpod/entrypoint.sh"]
CMD ["python", "app/app.py"]
```

Keep dependency installation before all application source copies so source edits reuse the expensive FlashAttention/application dependency layers. The official RunPod base supplies the matching CUDA toolkit, PyTorch, Triton, and `uv`; the build must print their versions and must not replace them with pip-installed CUDA/PyTorch packages.

- [ ] **Step 2: Add the entrypoint's persistent-path and SSH behavior**

Create `runpod/entrypoint.sh` with `set -eu` and these exact behaviors:

1. Set `DATA_ROOT=${ALEXANDRIA_DATA_ROOT:-/workspace/alexandria}`.
2. Derive and export `ALEXANDRIA_CONFIG_PATH`, `HF_HOME`, `TRANSFORMERS_CACHE`, and `TORCHINDUCTOR_CACHE_DIR` beneath the persistent root unless explicitly supplied.
3. Create directories for `config`, `uploads`, `scripts`, `designed_voices`, `clone_voices`, `lora_models`, `lora_datasets`, `dataset_builder`, `voicelines`, `preparer_output`, `logs`, `huggingface-cache`, and `torchinductor-cache`.
4. Replace the corresponding empty image directories with symlinks into `DATA_ROOT`; also symlink root-level runtime files (`annotated_script.json`, `chunks.json`, `state.json`, `voice_config.json`, `cloned_audiobook.mp3`, `audiobook.m4b`, `audacity_export.zip`, and `m4b_cover.jpg`) into `DATA_ROOT` so existing application paths persist without an application-wide refactor.
5. Copy `/alexandria/runpod/config.example.json` to the derived config path only when no persisted config exists.
6. Require a non-empty `PUBLIC_KEY`; write the exact single-line value to `/root/.ssh/authorized_keys` with directory mode 700 and file mode 600. Never log the key.
7. Create `/etc/ssh/sshd_config.d`, then write `/etc/ssh/sshd_config.d/alexandria.conf` with `PasswordAuthentication no`, `KbdInteractiveAuthentication no`, `PubkeyAuthentication yes`, `PermitRootLogin prohibit-password`, and `UsePAM no`.
8. Run `mkdir -p /run/sshd`, `ssh-keygen -A`, `sshd -t`, and `/usr/sbin/sshd`.
9. `exec "$@"` so Alexandria remains the foreground container process.

If `PUBLIC_KEY` is missing or `sshd -t` fails, exit with a clear error before starting Alexandria. Do not accept passwords or copy a private key into the container.

- [ ] **Step 3: Add the persisted POC configuration seed**

Create `runpod/config.example.json` with no translation model and valid `AppConfig` sections:

```json
{
  "llm": {
    "base_url": "http://127.0.0.1:11434/v1",
    "api_key": "local",
    "model_name": "unused-for-single-speaker-poc"
  },
  "tts": {
    "mode": "local",
    "url": "http://127.0.0.1:7860",
    "device": "auto",
    "language": "Spanish",
    "parallel_workers": 1,
    "compile_codec": false,
    "sub_batch_enabled": true,
    "sub_batch_max_items": 0
  }
}
```

The single-speaker UI path bypasses the LLM, so this placeholder model name is not invoked during the POC. Users can edit the persisted config or use the UI later.

- [ ] **Step 4: Add the GPU smoke test**

Create `runpod/smoke_test.py` with runtime imports and one sentence. It must:

```python
import os

import soundfile as sf
import torch
import triton
import flash_attn
from qwen_tts import Qwen3TTSModel

assert torch.cuda.is_available(), "CUDA is unavailable"
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"triton={triton.__version__}")
print(f"flash_attn={getattr(flash_attn, '__version__', 'installed')}")

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
wavs, sample_rate = model.generate_custom_voice(
    text="Este es un texto breve para verificar la síntesis de voz.",
    language="Spanish",
    speaker="Ryan",
    instruct="Neutral narration.",
    non_streaming_mode=True,
    max_new_tokens=512,
)
assert wavs and sample_rate > 0
output_root = os.environ.get("ALEXANDRIA_DATA_ROOT", "/tmp")
os.makedirs(output_root, exist_ok=True)
output_path = os.path.join(output_root, "alexandria-smoke.wav")
sf.write(output_path, wavs[0], sample_rate)
print(f"generated={len(wavs[0]) / sample_rate:.2f}s path={output_path}")
```

Do not run this during the CPU-only image build; run it after the image is attached to a GPU Pod.

- [ ] **Step 5: Validate the image's non-GPU dependency layer**

Build and inspect the image without invoking its SSH entrypoint:

```bash
cd /home/julia/devel/alexandria-audiobook
docker build --progress=plain -f Dockerfile.runpod -t alexandria-audiobook:runpod-test .
docker run --rm --entrypoint python alexandria-audiobook:runpod-test -c \
  'import torch, triton, flash_attn; print(torch.__version__, torch.version.cuda, triton.__version__, getattr(flash_attn, "__version__", "installed"))'
```

Expected: the build succeeds, the dependency layer imports succeed, and the printed Torch/CUDA/Triton versions match the pinned image. `torch.cuda.is_available()` is not required during this build-host check.

- [ ] **Step 6: Commit the image and runtime files**

```bash
cd /home/julia/devel/alexandria-audiobook
git add Dockerfile.runpod runpod/
git commit -m "feat: add RunPod FlashAttention image"
```

---

### Task 3: Add GHCR Publishing and RunPod Documentation

**Files:**
- Create: `.github/workflows/runpod-image.yml`
- Create: `runpod/.env.example`
- Create: `runpod/README.md`

**Interfaces:**
- Consumes: `Dockerfile.runpod`, `runpod/entrypoint.sh`, the fork `joaofauvel/alexandria-audiobook`, and GitHub's `GITHUB_TOKEN` package permissions.
- Produces: immutable GHCR tags and a reproducible manual Pod/key/UI/SCP procedure.

- [ ] **Step 1: Write the manual workflow definition**

Create `.github/workflows/runpod-image.yml` with `workflow_dispatch` and version-tag triggers only:

```yaml
name: Build RunPod image

on:
  workflow_dispatch:
  push:
    tags:
      - "v*"

permissions:
  contents: read
  packages: write

jobs:
  image:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ghcr.io/${{ github.repository_owner }}/alexandria-audiobook
          flavor: |
            latest=false
          tags: |
            type=sha,prefix=sha-
            type=ref,event=tag
      - uses: docker/build-push-action@v6
        with:
          context: .
          file: Dockerfile.runpod
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

Do not add a push-on-every-branch trigger or a `latest` tag in the first version.

- [ ] **Step 2: Add the minimal environment example**

Create `runpod/.env.example`:

```env
ALEXANDRIA_DATA_ROOT=/workspace/alexandria
```

Document that `PUBLIC_KEY` is supplied as a Pod environment value from the exact contents of the registered public key and is never committed to this file. Keep TTS settings in `/workspace/alexandria/config/config.json`.

- [ ] **Step 3: Document fork and image publishing**

In `runpod/README.md`, document these exact commands:

```bash
gh workflow run runpod-image.yml \
  --repo joaofauvel/alexandria-audiobook \
  --ref runpod-flashattention

gh run list --repo joaofauvel/alexandria-audiobook --workflow runpod-image.yml
gh run watch --repo joaofauvel/alexandria-audiobook
```

Explain that the workflow emits a `sha-` tag containing the commit's short hash or a version tag from a `v*` ref. Set a local `IMAGE_REF` variable to the exact emitted image reference and deploy that immutable SHA tag or digest, never `latest`.

- [ ] **Step 4: Document SSH key setup and verification**

Include this procedure:

```bash
install -d -m 700 "$HOME/.ssh"
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
  ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519" -C "runpod-alexandria"
fi
chmod 600 "$HOME/.ssh/id_ed25519"
chmod 644 "$HOME/.ssh/id_ed25519.pub"
runpodctl ssh add-key --key-file "$HOME/.ssh/id_ed25519.pub"
runpodctl ssh list-keys
```

State explicitly that RunPod must receive the actual `ssh-ed25519 ...` public-key line, not the `SHA256:` fingerprint. The Pod environment value is populated without displaying the private key. The Pod exposes `4200/http` and `22/tcp`, has a public IP, and sets `PUBLIC_KEY` to the public-key file contents.

- [ ] **Step 5: Document Pod creation and UI/SCP file flow**

Document the Pod settings:

```bash
export IMAGE_REF=ghcr.io/joaofauvel/alexandria-audiobook:sha-$(git rev-parse --short HEAD)
```

Use that value for the Pod image:

```text
image: $IMAGE_REF
GPU: NVIDIA RTX 3090 or RTX 4090 class, 24 GB VRAM preferred
volume mount: /workspace
ports: 4200/http, 22/tcp
environment: ALEXANDRIA_DATA_ROOT=/workspace/alexandria
             PUBLIC_KEY=$(cat "$HOME/.ssh/id_ed25519.pub")
```

Then document:

```bash
runpodctl ssh info "$POD_ID"
ssh -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -p "$SSH_PORT" \
  root@"$SSH_HOST" 'ss -lnt | grep -E ":22 |:4200 " && test -f /root/.ssh/authorized_keys'
scp -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -P "$SSH_PORT" \
  book.epub root@"$SSH_HOST":/workspace/alexandria/uploads/
scp -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -P "$SSH_PORT" \
  root@"$SSH_HOST":/workspace/alexandria/voicelines/audiobook.m4b ./audiobook.m4b
```

Explain that the UI is the normal path: open the RunPod HTTP proxy for port 4200, upload the EPUB/TXT, enable single-speaker Spanish narration, generate, export, and download. SCP/rsync is the bulk path; `runpodctl send/receive` is optional.

- [ ] **Step 6: Commit workflow and documentation**

```bash
cd /home/julia/devel/alexandria-audiobook
git add .github/workflows/runpod-image.yml runpod/.env.example runpod/README.md
git commit -m "docs: publish and deploy RunPod image"
```

---

### Task 4: Publish, Deploy, and Verify the End-to-End POC

**Files:**
- Modify: no source files; use the committed image and RunPod resources.
- Test artifact: `/workspace/alexandria/smoke-test.wav` and a small transfer fixture outside the repository.

**Interfaces:**
- Consumes: the GHCR image tag, RunPod Pod configuration, registered public key, and persisted volume.
- Produces: verified CUDA/FlashAttention/Triton runtime, verified SSH/SCP, reachable UI, and a generated Spanish audiobook sample.

- [ ] **Step 1: Push the implementation branch**

```bash
cd /home/julia/devel/alexandria-audiobook
git push origin runpod-flashattention
```

- [ ] **Step 2: Run the existing local Python checks**

```bash
cd /home/julia/devel/alexandria-audiobook/app
python -m unittest test_tts -v
python -m py_compile tts.py test_tts.py
```

Expected: the focused tests pass and compilation succeeds. Do not run Pinokio or alter its files as part of this RunPod validation.

- [ ] **Step 3: Trigger and watch the GHCR build**

```bash
gh workflow run runpod-image.yml \
  --repo joaofauvel/alexandria-audiobook \
  --ref runpod-flashattention
gh run watch --repo joaofauvel/alexandria-audiobook
```

Expected: the workflow succeeds and publishes a SHA-tagged image. Set `IMAGE_REF` to the exact tag/digest from the successful run; do not deploy `latest`.

- [ ] **Step 4: Register and verify the local public key before Pod creation**

```bash
runpodctl ssh add-key --key-file "$HOME/.ssh/id_ed25519.pub"
runpodctl ssh list-keys
ssh-keygen -lf "$HOME/.ssh/id_ed25519.pub"
```

Compare only the public-key fingerprint/listing. Never print or upload `$HOME/.ssh/id_ed25519`.

- [ ] **Step 5: Create the Pod with both ports and persistent storage**

Use RunPod MCP or the console to create one Pod from the immutable GHCR image with a 24 GB-class GPU, a public IP, `/workspace` volume mount, ports `4200/http` and `22/tcp`, `ALEXANDRIA_DATA_ROOT=/workspace/alexandria`, and `PUBLIC_KEY` equal to the exact public-key line from `$HOME/.ssh/id_ed25519.pub`.

Do not create a Serverless endpoint. Record the Pod ID, SSH host, mapped SSH port, and HTTP proxy URL.

- [ ] **Step 6: Verify SSH and SCP before processing a book**

Obtain the connection details:

```bash
runpodctl ssh info "$POD_ID"
```

Set the returned host and mapped port locally without committing them. The following interactive read accepts the two values printed by `runpodctl ssh info`:

```bash
read -r SSH_HOST SSH_PORT
```

Verify the daemon, key-only login, and ports:

```bash
ssh -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -p "$SSH_PORT" \
  root@"$SSH_HOST" \
  'ss -lnt | grep -E ":22 |:4200 " && test -s /root/.ssh/authorized_keys'
```

Verify a file round trip:

```bash
printf 'alexandria transfer check\n' > /tmp/alexandria-transfer.txt
scp -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -P "$SSH_PORT" \
  /tmp/alexandria-transfer.txt root@"$SSH_HOST":/workspace/alexandria/
ssh -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -p "$SSH_PORT" \
  root@"$SSH_HOST" 'cat /workspace/alexandria/alexandria-transfer.txt'
scp -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -P "$SSH_PORT" \
  root@"$SSH_HOST":/workspace/alexandria/alexandria-transfer.txt \
  /tmp/alexandria-transfer-returned.txt
cmp /tmp/alexandria-transfer.txt /tmp/alexandria-transfer-returned.txt
```

Expected: SSH uses the registered key without a password, both ports listen, and `cmp` exits successfully.

- [ ] **Step 7: Run the GPU smoke test on the Pod**

```bash
ssh -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -p "$SSH_PORT" \
  root@"$SSH_HOST" 'python /alexandria/runpod/smoke_test.py'
```

Expected: CUDA is available, `flash_attn` and Triton import, the 1.7B CustomVoice model loads with `flash_attention_2`, and a Spanish WAV is generated. If this fails, stop before enabling codec compilation or running a full book.

- [ ] **Step 8: Verify the Alexandria UI and single-speaker export**

Open the recorded HTTP proxy URL for port 4200. In the UI:

1. Confirm TTS mode is `local` and language is `Spanish`.
2. Upload a small Spanish TXT/EPUB.
3. Enable single-speaker mode with speaker `Narrator`.
4. Use `Neutral narration.` as the instruction.
5. Generate one or a few chunks.
6. Export the audiobook and download the result.

Expected: the UI remains responsive, chunks reach `done`, and the downloaded audio is non-empty and playable.

- [ ] **Step 9: Verify persistence**

Record the generated output path and config path, restart the Pod, reconnect over SSH, and verify:

```bash
ssh -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -p "$SSH_PORT" \
  root@"$SSH_HOST" \
  'test -s /workspace/alexandria/config/config.json && \
   test -d /workspace/alexandria/huggingface-cache && \
   find /workspace/alexandria/voicelines -type f -size +0c | head -1'
```

Expected: configuration, cache, and generated audio remain available without rebuilding the image.

- [ ] **Step 10: Commit verified deployment notes**

After replacing any runtime-only values with general instructions and omitting private data, record the verified image tag, port/key requirements, and smoke-test result in `runpod/README.md`, then commit:

```bash
cd /home/julia/devel/alexandria-audiobook
git add runpod/README.md
git commit -m "docs: record RunPod verification steps"
git push origin runpod-flashattention
```

Do not commit Pod IDs, public IPs, mapped ports, public keys, private keys, credentials, or generated user media.

---

## Final Verification Checklist

- [ ] Only the RunPod feature files and `app/tts.py` changed; Pinokio files remain byte-for-byte unchanged.
- [ ] `python -m unittest test_tts -v` passes.
- [ ] Docker dependency import check passes.
- [ ] GHCR workflow publishes a SHA-tagged image with cache enabled.
- [ ] Pod exposes both `4200/http` and `22/tcp`.
- [ ] Exact public key is registered and authorized; private key remains local.
- [ ] SSH and SCP round trip pass with `IdentitiesOnly=yes`.
- [ ] CUDA, FlashAttention, and Triton smoke test passes on the Pod.
- [ ] Spanish single-speaker UI generation and download pass.
- [ ] Persistent volume retains config, model cache, state, and output after restart.
- [ ] No translation, Serverless, remote Gradio, or Pinokio changes were introduced.
