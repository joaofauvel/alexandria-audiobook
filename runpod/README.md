# Alexandria on RunPod

This deployment is for the RunPod-specific fork branch. It leaves the existing Pinokio scripts and local `Dockerfile` unchanged.

## Image contract

`Dockerfile.runpod` starts from the pinned official RunPod image:

```text
runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404
```

That base supplies CUDA 12.8, PyTorch 2.8, Triton 3.4, `nvcc`, `uv`, FFmpeg, SSH tooling, and a default `HF_HOME=/workspace/.cache/huggingface/`. Alexandria does **not** install a second CUDA or PyTorch stack. Application dependencies and `flash-attn==2.8.3` are installed with `uv pip`; the Ubuntu base is externally managed, so the Dockerfile explicitly uses `--break-system-packages`. The entrypoint preserves this explicit RunPod cache location, uses its `/hub` subdirectory for `HF_HUB_CACHE` and `TRANSFORMERS_CACHE`, and keeps it on the mounted `/workspace` volume.

The image build checks Torch/CUDA, Triton, FlashAttention, and Qwen imports. The GPU-only smoke test also verifies that Qwen reports `flash_attention_2` and completes one Spanish CustomVoice generation.

References:

- [RunPod custom Dockerfiles](https://docs.runpod.io/tutorials/introduction/containers/create-dockerfiles)
- [RunPod storage types](https://docs.runpod.io/pods/storage/types)
- [RunPod SSH connections](https://docs.runpod.io/pods/configuration/use-ssh)
- [RunPod PyTorch images](https://hub.docker.com/r/runpod/pytorch/tags)
- [uv Docker cache mounts](https://docs.astral.sh/uv/guides/integration/docker/)

## Local checks

The image build requires Docker BuildKit and compiles FlashAttention. It does not require a host GPU for the import layer:

```bash
docker build --progress=plain \
  -f Dockerfile.runpod \
  -t alexandria-audiobook:runpod-test \
  .

docker run --rm --entrypoint python alexandria-audiobook:runpod-test -c \
  'import torch, triton, flash_attn, qwen_tts; print(torch.__version__, torch.version.cuda, triton.__version__, getattr(flash_attn, "__version__", "installed"))'
```

The default entrypoint requires `PUBLIC_KEY`, so dependency-only checks override it with `--entrypoint python`.

For a local GPU smoke test, pass the registered public key and use the NVIDIA runtime explicitly when Docker's `--gpus all` selects `runc` on the host:

```bash
export PUBLIC_KEY="$(cat "$HOME/.ssh/id_ed25519.pub")"
docker run --rm --runtime=nvidia \
  -e PUBLIC_KEY \
  -v "$PWD/.runpod-local-workspace:/workspace" \
  alexandria-audiobook:runpod-test \
  python /alexandria/runpod/smoke_test.py
```

## Publish to GHCR

The workflow runs only on manual dispatch and version tags. It uses GitHub Actions cache for the dependency and FlashAttention layers:

```bash
gh workflow run runpod-image.yml \
  --repo joaofauvel/alexandria-audiobook \
  --ref runpod-flashattention

gh run list \
  --repo joaofauvel/alexandria-audiobook \
  --workflow runpod-image.yml
```

Watch the newest run and use its SHA tag:

```bash
gh run watch --repo joaofauvel/alexandria-audiobook
export IMAGE_REF="ghcr.io/joaofauvel/alexandria-audiobook:sha-$(git rev-parse --short HEAD)"
```

Use the exact tag/digest emitted by the successful run; do not deploy `latest`. After the first push, make the GHCR package public or configure RunPod with a read-only GHCR registry credential. Never commit a registry token.

## SSH key setup

Generate an Ed25519 key locally if needed and register only its public half with RunPod:

```bash
install -d -m 700 "$HOME/.ssh"
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
  ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519" -C "runpod-alexandria"
fi
chmod 600 "$HOME/.ssh/id_ed25519"
chmod 644 "$HOME/.ssh/id_ed25519.pub"
runpodctl ssh add-key --key-file "$HOME/.ssh/id_ed25519.pub"
runpodctl ssh list-keys
ssh-keygen -lf "$HOME/.ssh/id_ed25519.pub"
```

RunPod must receive the complete `ssh-ed25519 ...` line, not its `SHA256:` fingerprint. The private key stays on the local machine.

## Create the Pod

Create a regular Pod, not a Serverless endpoint, from `$IMAGE_REF` with:

```text
GPU: NVIDIA RTX 3090 or RTX 4090 class, 24 GB VRAM preferred
container disk: 40 GB or larger
volume disk: 50 GB or larger mounted at /workspace
ports: 4200/http, 22/tcp
ALEXANDRIA_DATA_ROOT: /workspace/alexandria
PUBLIC_KEY: the exact contents of ~/.ssh/id_ed25519.pub
```

A RunPod volume disk survives Pod stop/restart and is deleted with the Pod. Use a network volume instead when data must survive Pod deletion or move between Pods. The container disk is for the image and temporary OS files, not book data or model caches.

The entrypoint creates `/workspace/alexandria`, persists Alexandria state and audio there, and starts SSH before Alexandria. It rejects startup when `PUBLIC_KEY` is missing, disables password authentication, and permits root login only with the supplied public key.

## Verification record

Local verification completed against `alexandria-audiobook:runpod-test`:

- Pinned Torch 2.8.0+cu128, CUDA 12.8, Triton 3.4.0, FlashAttention 2.8.3, and Qwen TTS loaded on an RTX 4060.
- The smoke test reported `attn_implementation=flash_attention_2` and generated Spanish audio in 6.16 seconds.
- The API path passed Spanish configuration, TXT upload, single-speaker script generation, chunk audio, Audacity ZIP export, and M4B export.
- Recreating the container with the same `/workspace` volume preserved config, script, audio, model cache, and exports; key-only SSH and SCP round-trip passed.
- The published immutable image is `ghcr.io/joaofauvel/alexandria-audiobook:sha-fb5116a` with digest `sha256:380c0a2dc8468aee50705f596dd780575aa4f90c416faee2a0a2bfe2a9402c24`.

A RunPod 3090 deployment was attempted with the authenticated GHCR credential and the final image, but the Pod remained `runtime=null` with `uptime=0` while waiting for host scheduling and was deleted to avoid idle billing. The existing 3090 POC Pod remains stopped (`EXITED`); no GPU Pod is currently running.

## SSH and SCP verification

Get the Pod-specific host and mapped SSH port:

```bash
export POD_ID="pod-id-from-runpod"
runpodctl ssh info "$POD_ID"
```

Set the two values printed by that command locally without committing them:

```bash
read -r SSH_HOST SSH_PORT
```

Verify key-only SSH and both container listeners:

```bash
ssh -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519" \
  -p "$SSH_PORT" \
  root@"$SSH_HOST" \
  'ss -lnt | grep -E ":22 |:4200 " && test -s /root/.ssh/authorized_keys'
```

Verify a file round trip before processing a book:

```bash
printf 'alexandria transfer check\n' > /tmp/alexandria-transfer.txt
scp -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519" \
  -P "$SSH_PORT" \
  /tmp/alexandria-transfer.txt \
  root@"$SSH_HOST":/workspace/alexandria/

ssh -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519" \
  -p "$SSH_PORT" \
  root@"$SSH_HOST" \
  'cat /workspace/alexandria/alexandria-transfer.txt'

scp -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519" \
  -P "$SSH_PORT" \
  root@"$SSH_HOST":/workspace/alexandria/alexandria-transfer.txt \
  /tmp/alexandria-transfer-returned.txt

cmp /tmp/alexandria-transfer.txt /tmp/alexandria-transfer-returned.txt
```

`runpodctl send`/`receive` remains an optional one-off transfer path, but SSH/SCP is the required verified bulk path.

## GPU smoke test

Run this only after the image is attached to a GPU Pod:

```bash
ssh -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519" \
  -p "$SSH_PORT" \
  root@"$SSH_HOST" \
  'python /alexandria/runpod/smoke_test.py'
```

The test reports CUDA, GPU, Triton, and FlashAttention versions; loads `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` with `flash_attention_2`; generates a short Spanish sample; and writes `alexandria-smoke.wav` under `/workspace/alexandria`.

Keep `compile_codec` disabled for this first check. Enable it later in the persisted config only after normal generation is stable.

## Alexandria UI workflow

Open the RunPod HTTP proxy for port 4200 and use the existing UI:

1. Confirm local TTS and Spanish in the persisted TTS configuration.
2. Upload a small Spanish TXT or EPUB.
3. Use single-speaker mode with the built-in `Ryan` voice or the UI's narrator preset.
4. Generate one or a few chunks with neutral narration.
5. Export the audiobook and download it through the UI.

The normal book path is the UI. Use SCP/rsync for bulk inputs and outputs.

## Persistence check

After a successful sample, stop and restart the Pod, reconnect, and verify that configuration, cache, state, and audio remain:

```bash
ssh -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519" \
  -p "$SSH_PORT" \
  root@"$SSH_HOST" \
  'test -s /workspace/alexandria/config/config.json && \
   test -d /workspace/.cache/huggingface && \
   find /workspace/alexandria/voicelines -type f -size +0c | head -1'
```
