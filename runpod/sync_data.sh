#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sync_data.sh push [POD_ID] [options]
  sync_data.sh pull [POD_ID] [options]

Sync Alexandria data between this machine and a RunPod Pod over SSH/rsync.

Options:
  --pod-id ID          Resolve SSH details with runpodctl (optional)
  --host HOST          SSH host (use when runpodctl cannot see a v2 Pod)
  --port PORT          SSH port
  --user USER          SSH user (default: root)
  --ssh-key PATH       SSH private key (default: ~/.ssh/id_ed25519)
  --local-root PATH    Root containing alexandria/ and .cache/huggingface/
  --local-data PATH    Alexandria data directory
  --local-cache PATH   Hugging Face cache directory
  --cache              Include the Hugging Face cache
  --dry-run            Print rsync commands without connecting
  -h, --help           Show this help

Defaults:
  local root:  ${ALEXANDRIA_SYNC_ROOT:-$HOME/.local/share/alexandria-audiobook}
  remote data: /workspace/alexandria
  remote cache: /workspace/.cache/huggingface

The sync never uses rsync --delete. It does not remove files from either side.
For v2 Pods that runpodctl cannot resolve, pass --host and --port explicitly.
EOF
}

die() {
    echo "sync_data.sh: $*" >&2
    exit 2
}

[[ $# -ge 1 ]] || { usage; exit 2; }
DIRECTION="$1"
shift
case "$DIRECTION" in
    push|pull) ;;
    -h|--help) usage; exit 0 ;;
    *) die "direction must be push or pull" ;;
esac

SYNC_ROOT="${ALEXANDRIA_SYNC_ROOT:-$HOME/.local/share/alexandria-audiobook}"
LOCAL_DATA="${ALEXANDRIA_LOCAL_DATA:-$SYNC_ROOT/alexandria}"
LOCAL_CACHE="${ALEXANDRIA_LOCAL_CACHE:-$SYNC_ROOT/.cache/huggingface}"
REMOTE_DATA="/workspace/alexandria"
REMOTE_CACHE="/workspace/.cache/huggingface"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_USER="${SSH_USER:-}"
SSH_HOST="${SSH_HOST:-}"
SSH_PORT="${SSH_PORT:-}"
POD_ID=""
INCLUDE_CACHE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pod-id)
            [[ $# -ge 2 ]] || die "--pod-id requires a value"
            POD_ID="$2"
            shift 2
            ;;
        --host)
            [[ $# -ge 2 ]] || die "--host requires a value"
            SSH_HOST="$2"
            shift 2
            ;;
        --port)
            [[ $# -ge 2 ]] || die "--port requires a value"
            SSH_PORT="$2"
            shift 2
            ;;
        --user)
            [[ $# -ge 2 ]] || die "--user requires a value"
            SSH_USER="$2"
            shift 2
            ;;
        --ssh-key)
            [[ $# -ge 2 ]] || die "--ssh-key requires a value"
            SSH_KEY="$2"
            shift 2
            ;;
        --local-root)
            [[ $# -ge 2 ]] || die "--local-root requires a value"
            SYNC_ROOT="$2"
            LOCAL_DATA="$SYNC_ROOT/alexandria"
            LOCAL_CACHE="$SYNC_ROOT/.cache/huggingface"
            shift 2
            ;;
        --local-data)
            [[ $# -ge 2 ]] || die "--local-data requires a value"
            LOCAL_DATA="$2"
            shift 2
            ;;
        --local-cache)
            [[ $# -ge 2 ]] || die "--local-cache requires a value"
            LOCAL_CACHE="$2"
            shift 2
            ;;
        --cache)
            INCLUDE_CACHE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

if [[ -n "$POD_ID" && ( -z "$SSH_HOST" || -z "$SSH_PORT" ) ]]; then
    command -v runpodctl >/dev/null 2>&1 || die "runpodctl is required with --pod-id"
    INFO="$(runpodctl ssh info "$POD_ID" -o json 2>/dev/null || true)"
    if [[ -n "$INFO" ]]; then
        ENDPOINT="$(INFO="$INFO" python3 - <<'PY'
import json
import os

raw = os.environ.get("INFO", "")
objects = []
for line in raw.splitlines():
    try:
        objects.append(json.loads(line))
    except json.JSONDecodeError:
        continue

candidates = []
def visit(value, path=()):
    if isinstance(value, dict):
        if isinstance(value.get("host"), str) and value.get("port") is not None:
            candidates.append((".".join(path), value))
        for key, child in value.items():
            visit(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            visit(child, path + (str(index),))

for obj in objects:
    visit(obj)

# Prefer a direct endpoint over a proxy endpoint.
candidates.sort(key=lambda item: ("proxy" in item[0].lower(), item[0]))
if candidates:
    value = candidates[0][1]
    print(f'{value["host"]}\t{value["port"]}\t{value.get("username", "root")}')
PY
)"
        if [[ -n "$ENDPOINT" ]]; then
            IFS=$'\t' read -r RESOLVED_HOST RESOLVED_PORT RESOLVED_USER <<< "$ENDPOINT"
            SSH_HOST="${SSH_HOST:-$RESOLVED_HOST}"
            SSH_PORT="${SSH_PORT:-$RESOLVED_PORT}"
            SSH_USER="${SSH_USER:-$RESOLVED_USER}"
        fi
    fi
fi

SSH_USER="${SSH_USER:-root}"
[[ -n "$SSH_HOST" ]] || die "SSH host is required; pass --host or a resolvable --pod-id"
[[ -n "$SSH_PORT" ]] || die "SSH port is required; pass --port or a resolvable --pod-id"
[[ -f "$SSH_KEY" ]] || die "SSH key not found: $SSH_KEY"
command -v rsync >/dev/null 2>&1 || die "rsync is required"

if [[ "$DIRECTION" == "push" && ! -d "$LOCAL_DATA" ]]; then
    die "local Alexandria data directory not found: $LOCAL_DATA"
fi
if [[ "$INCLUDE_CACHE" == 1 && "$DIRECTION" == "push" && ! -d "$LOCAL_CACHE" ]]; then
    die "local Hugging Face cache directory not found: $LOCAL_CACHE"
fi

REMOTE="$SSH_USER@$SSH_HOST"
printf -v RSYNC_SSH 'ssh -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -i %q -p %q' "$SSH_KEY" "$SSH_PORT"

run_rsync() {
    local source="$1"
    local destination="$2"
    local command=(rsync -aH --info=progress2 -e "$RSYNC_SSH" "$source" "$destination")

    if [[ "$DRY_RUN" == 1 ]]; then
        printf '+ '
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
}

prepare_remote_dir() {
    local directory="$1"
    local quoted_directory
    printf -v quoted_directory '%q' "$directory"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '+ ssh -o IdentitiesOnly=yes -i %q -p %q %q mkdir -p -- %s\n' \
            "$SSH_KEY" "$SSH_PORT" "$REMOTE" "$quoted_directory"
    else
        ssh -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
            -i "$SSH_KEY" -p "$SSH_PORT" "$REMOTE" \
            "mkdir -p -- $quoted_directory"
    fi
}

if [[ "$DIRECTION" == "push" ]]; then
    prepare_remote_dir "$REMOTE_DATA"
    run_rsync "$LOCAL_DATA/" "$REMOTE:$REMOTE_DATA/"
    if [[ "$INCLUDE_CACHE" == 1 ]]; then
        prepare_remote_dir "$REMOTE_CACHE"
        run_rsync "$LOCAL_CACHE/" "$REMOTE:$REMOTE_CACHE/"
    fi
else
    mkdir -p "$LOCAL_DATA"
    run_rsync "$REMOTE:$REMOTE_DATA/" "$LOCAL_DATA/"
    if [[ "$INCLUDE_CACHE" == 1 ]]; then
        mkdir -p "$LOCAL_CACHE"
        run_rsync "$REMOTE:$REMOTE_CACHE/" "$LOCAL_CACHE/"
    fi
fi

echo "sync complete: $DIRECTION"
