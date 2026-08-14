# Alexandria RunPod Deployment Design

## Status

Approved design for review before implementation.

## Goal

Create a reproducible, RunPod-specific deployment for Alexandria Audiobook Generator that uses Qwen3-TTS with FlashAttention 2 on NVIDIA CUDA, while leaving the existing local Pinokio installation path unchanged.

The first proof of concept targets Spanish narration with Alexandria's existing single-speaker workflow and built-in voice presets. It does not add translation or a remote TTS adapter.

## Scope

### In scope

- A RunPod-specific Docker image built from the fork `joaofauvel/alexandria-audiobook`.
- Explicit Qwen model loading with `attn_implementation="flash_attention_2"`.
- CUDA, FlashAttention, and Triton build/runtime checks.
- GitHub Actions image publishing to GHCR with cached layers.
- A persistent RunPod volume for Alexandria state, model cache, configuration, and audio.
- A minimal RunPod entrypoint that maps Alexandria's existing runtime paths into the persistent volume.
- A RunPod deployment guide using the Alexandria web UI as the primary workflow and SSH/SCP as the verified bulk-transfer path.
- An SSH daemon in the RunPod image with public-key-only authentication.
- A runtime smoke test that performs a short Spanish generation and an SSH/SCP transfer check.

### Out of scope

- Changes to Pinokio scripts or the existing local Docker path.
- Translation or TranslateGemma integration.
- Serverless endpoints.
- A remote Gradio TTS adapter.
- Voice cloning, VoiceDesign, or LoRA changes.
- Automatic RunPod provisioning through a committed CLI/MCP script.
- A new application-wide data-root abstraction.
- A 0.6B model fallback unless the smoke test demonstrates that the 1.7B model cannot run on the selected hardware.

## Architecture

### Build artifact

The image will be built from a new RunPod-specific Dockerfile. Dependency layers will precede application source layers:

1. PyTorch CUDA 12.8 development base.
2. OS audio/build dependencies.
3. Pinned Python requirements.
4. Pinned `qwen-tts` and FlashAttention build.
5. Alexandria source and deployment files.

The initial build uses:

- PyTorch 2.8 CUDA 12.8.
- `qwen-tts==0.1.1`.
- `flash-attn==2.8.3`, compiled with `--no-build-isolation`.
- `MAX_JOBS=2` to limit build memory pressure.
- `TORCH_CUDA_ARCH_LIST="8.6;8.9"` for RTX 3090/4090-class RunPod GPUs and the local RTX 4060 if the image is tested locally.

The final image may retain the development CUDA toolkit for the first iteration. Reducing image size is not part of the initial milestone.

### FlashAttention enforcement

On NVIDIA, FlashAttention 2 uses CUDA kernels. Triton is verified separately as the PyTorch-matched compiler/runtime; the AMD-specific Triton FlashAttention environment path is not enabled for NVIDIA.

The application change is limited to `app/tts.py`: every local Qwen model load (CustomVoice, Base, VoiceDesign, and LoRA base) passes:

```python
attn_implementation="flash_attention_2"
```

The image smoke test must verify:

- `torch.cuda.is_available()` is true.
- the expected CUDA runtime is available.
- `flash_attn` imports.
- `triton` imports.
- a Qwen CustomVoice model loads with FlashAttention 2.
- one short Spanish generation completes.

The application must not silently fall back to eager/SDPA attention when FlashAttention is required.

### Runtime storage

The Pod mounts persistent storage at `/workspace`. The RunPod entrypoint creates a data root beneath that mount and maps the existing Alexandria runtime directories into it without changing the application path model.

Persistent data includes:

- uploaded books;
- `config.json` and project state files;
- generated audio and exports;
- saved scripts and voice configuration;
- Hugging Face model cache;
- TorchInductor compilation cache.

TTS settings remain in the persisted Alexandria configuration rather than being duplicated in environment variables. The initial POC configuration uses local TTS, Spanish, a built-in voice, and single-speaker mode.

The required deployment values are the persistent data root and the registered public SSH key, with derived paths supplied by the entrypoint. A committed example environment file will not contain translation or model-selection settings.

### Web UI and file flow

The Pod exposes Alexandria's existing HTTP service on internal port `4200` and binds the service to `0.0.0.0`.

The primary workflow is:

1. Open the RunPod HTTP proxy URL for port 4200.
2. Upload an EPUB/TXT book through Alexandria's UI.
3. Enable single-speaker mode and select Spanish narration settings.
4. Generate audio and export the audiobook.
5. Download the result through the UI.

The image also exposes TCP port 22 for verified bulk transfers. The entrypoint must:

- include `openssh-server` in the Docker image;
- generate host keys at startup with `ssh-keygen -A`;
- write the RunPod-provided `PUBLIC_KEY` value to `/root/.ssh/authorized_keys` with mode 600;
- disable password authentication and permit public-key authentication only;
- start `sshd` before starting Alexandria.

The Pod must have a public IP for full SSH/SCP/SFTP access. The private key remains on the local machine and is never copied into the image or Pod. `runpodctl send/receive` remains an optional alternative for one-off transfers.

## Image publishing and deployment

GitHub Actions runs only on manual dispatch and version tags, not every commit. It builds the RunPod Dockerfile with BuildKit and publishes:

```text
ghcr.io/joaofauvel/alexandria-audiobook:<commit-or-release-tag>
```

The workflow uses GitHub Actions layer caching so dependency and FlashAttention layers are reused. RunPod deployments use an immutable commit/release tag or image digest rather than `latest`.

A Pod is created manually through RunPod's console, MCP, or `runpodctl` with:

- a 24 GB-class NVIDIA GPU for the throughput test;
- the GHCR image;
- a persistent volume mounted at `/workspace`;
- a public IP suitable for full SSH access;
- ports `4200/http` and `22/tcp` exposed;
- `PUBLIC_KEY` set to the exact contents of the registered local public key.

Before connecting, the deployment procedure registers the public key with RunPod using `runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub` (or the equivalent account setting), verifies it with `runpodctl ssh list-keys`, and obtains the Pod-specific SSH command with `runpodctl ssh info <pod-id>`.

The initial milestone does not create or delete cloud resources automatically.

## Codec compilation and performance

Codec compilation is independent of FlashAttention. Alexandria applies `torch.compile` to the Qwen speech-tokenizer decoder to improve repeated batch decoding after an initial compilation/warmup cost.

The first smoke test keeps `compile_codec` disabled. Once FlashAttention and normal generation are verified, the setting can be enabled in the persisted RunPod `config.json` without rebuilding the image. The TorchInductor cache is persisted on the volume.

Batch sizing starts with Alexandria's existing VRAM-aware automatic sub-batching. A fixed 40–60 item batch is not assumed until a representative benchmark demonstrates that it is stable on the selected GPU.

## Verification and acceptance criteria

The implementation is accepted when all of the following are true:

1. The fork contains the RunPod-specific files and the focused `app/tts.py` change; Pinokio files are unchanged.
2. A GitHub Actions run builds and publishes the image to GHCR using cached dependency layers.
3. The smoke test passes on a GPU-backed environment and reports CUDA, FlashAttention, and Triton availability.
4. The smoke test performs a Spanish CustomVoice generation using FlashAttention 2.
5. The Alexandria UI is reachable through the RunPod HTTP proxy.
6. A Spanish text/EPUB can be uploaded through the UI, processed in single-speaker mode, and exported/downloaded.
7. An SSH connection succeeds using the registered private key with `IdentitiesOnly=yes`, and a sample input/output file round trip succeeds with SCP.
8. Restarting the Pod/container preserves configuration, model cache, project state, and generated output through the mounted volume.
9. Changing persisted configuration does not require rebuilding the image.

## Risks and mitigations

- **FlashAttention build compatibility:** compile against the pinned PyTorch/CUDA base and fail the image smoke test if the extension or model load fails.
- **VRAM pressure on local hardware:** local Pinokio remains the primary local path; the RunPod acceptance target uses a 24 GB-class GPU.
- **RunPod volume path assumptions:** keep the path mapping isolated in the RunPod entrypoint rather than changing Alexandria's application-wide storage model.
- **SSH setup errors:** register the exact public-key contents (not its fingerprint), require a public IP and TCP port 22, disable password authentication, and test SSH/SCP before processing a book.
- **Large file transfer limitations:** use the UI for normal books and verified SSH/rsync or `runpodctl` for bulk transfers.
- **Slow rebuilds:** keep code layers last and use BuildKit/GitHub Actions cache; configuration and model weights remain outside the image.
