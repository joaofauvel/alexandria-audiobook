#!/bin/sh
set -eu

DATA_ROOT="${ALEXANDRIA_DATA_ROOT:-/workspace/alexandria}"
export ALEXANDRIA_DATA_ROOT="$DATA_ROOT"
CONFIG_DIR="$DATA_ROOT/config"

export ALEXANDRIA_CONFIG_PATH="${ALEXANDRIA_CONFIG_PATH:-$CONFIG_DIR/config.json}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/huggingface-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$DATA_ROOT/torchinductor-cache}"

mkdir -p \
    "$DATA_ROOT" \
    "$CONFIG_DIR" \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$(dirname "$ALEXANDRIA_CONFIG_PATH")"

for dir_name in \
    uploads \
    scripts \
    designed_voices \
    clone_voices \
    lora_models \
    lora_datasets \
    dataset_builder \
    voicelines \
    preparer_output \
    logs
 do
    mkdir -p "$DATA_ROOT/$dir_name"
done

link_dir() {
    target="$1"
    data_path="$DATA_ROOT/$2"

    if [ -L "$target" ] && [ "$(readlink "$target")" = "$data_path" ]; then
        return
    fi

    if [ -e "$target" ] || [ -L "$target" ]; then
        rm -rf "$target"
    fi
    ln -s "$data_path" "$target"
}

link_file() {
    target="$1"
    data_path="$DATA_ROOT/$2"

    if [ -L "$target" ] && [ "$(readlink "$target")" = "$data_path" ]; then
        return
    fi

    if [ -e "$target" ] || [ -L "$target" ]; then
        rm -rf "$target"
    fi
    ln -s "$data_path" "$target"
}

link_dir /alexandria/app/uploads uploads
link_dir /alexandria/scripts scripts
link_dir /alexandria/designed_voices designed_voices
link_dir /alexandria/clone_voices clone_voices
link_dir /alexandria/lora_models lora_models
link_dir /alexandria/lora_datasets lora_datasets
link_dir /alexandria/dataset_builder dataset_builder
link_dir /alexandria/voicelines voicelines
link_dir /alexandria/preparer_output preparer_output
link_dir /alexandria/logs logs

for file_name in \
    annotated_script.json \
    chunks.json \
    state.json \
    voice_config.json \
    cloned_audiobook.mp3 \
    audiobook.m4b \
    audacity_export.zip \
    m4b_cover.jpg
 do
    link_file "/alexandria/$file_name" "$file_name"
done

if [ ! -f "$ALEXANDRIA_CONFIG_PATH" ]; then
    cp /alexandria/runpod/config.example.json "$ALEXANDRIA_CONFIG_PATH"
fi

if [ -z "${PUBLIC_KEY:-}" ]; then
    echo "PUBLIC_KEY must contain the registered SSH public key" >&2
    exit 1
fi

install -d -m 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

mkdir -p /etc/ssh/sshd_config.d /run/sshd
cat > /etc/ssh/sshd_config.d/alexandria.conf <<'EOF'
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
PubkeyAuthentication yes
PermitRootLogin prohibit-password
AuthorizedKeysFile .ssh/authorized_keys
UsePAM no
EOF

ssh-keygen -A
sshd -t
/usr/sbin/sshd

exec "$@"
