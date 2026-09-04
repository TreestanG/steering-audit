#!/usr/bin/env bash
# Set up a fresh Ubuntu GPU box from this Mac: clone the repo over ssh, install deps, copy the HF token.
#
#   scripts/aws_bootstrap.sh ubuntu@HOST [box-key.pem]
#
# HOST      the box's user@address (the Ubuntu DLAMI user is ubuntu)
# key.pem   ssh key for the box, default ~/.ssh/tristan-west-2.pem
# GH_KEY    env var, GitHub ssh key to forward, default ~/.ssh/id_ed25519 (must be added on github.com)
#
# GitHub is reached through ssh-agent forwarding, so no key is copied to the box;
# a later `git pull` there needs `ssh -A`. The HF token comes from ~/.cache/huggingface/token.
set -eu
HOST=${1:?usage: aws_bootstrap.sh user@host [box-key.pem]}
KEY=${2:-$HOME/.ssh/tristan-west-2.pem}
GH_KEY=${GH_KEY:-$HOME/.ssh/id_ed25519}
OPTS=(-o StrictHostKeyChecking=accept-new)
[[ -f $KEY ]] && OPTS+=(-i "$KEY")

ssh-add -l > /dev/null 2>&1 || ssh-add "$GH_KEY"
ssh -o BatchMode=yes -T git@github.com 2>&1 | grep -q "successfully authenticated" \
  || { echo "GitHub rejects $GH_KEY: add $GH_KEY.pub at https://github.com/settings/keys"; exit 1; }

ssh "${OPTS[@]}" "$HOST" 'mkdir -p ~/.cache/huggingface && cat > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token' \
  < ~/.cache/huggingface/token

ssh -A "${OPTS[@]}" "$HOST" bash -s <<'REMOTE'
set -eu
export PATH="$HOME/.local/bin:$PATH"
DIR=$HOME/aat
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git tmux rsync > /dev/null
mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts 2> /dev/null
git config --global user.name "Tristan Gee"
git config --global user.email "tristangeea@gmail.com"
if [[ -d $DIR/.git ]]; then
  git -C "$DIR" pull --ff-only
else
  git clone git@github.com:TreestanG/reachability-auditing.git "$DIR"
fi
cd "$DIR"
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.__version__, torch.cuda.get_device_name(0))"
MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
KV=$(( (MEM_MB - 8000) * 1000000 / 2 ))
grep -q AAT_KV_BUDGET ~/.profile 2> /dev/null || echo "export AAT_KV_BUDGET=$KV" >> ~/.profile
echo "ready: $DIR  AAT_KV_BUDGET=$KV  (git pull on the box needs ssh -A)"
REMOTE
