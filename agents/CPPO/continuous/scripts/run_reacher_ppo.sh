#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

export WANDB_API_KEY=${WANDB_API_KEY:-}
export MUJOCO_GL=${MUJOCO_GL:-off}

if [ -f scripts/fix_execstack.py ]; then
    python scripts/fix_execstack.py >/dev/null 2>&1 || true
fi

python -u -m jaxgcrl.run \
    --config-name=ppo \
    env=reacher \
    run.log_wandb=true \
    run.wandb_project_name=cppo-jaxgcrl \
    "run.wandb_group=jaxgcrl-ppo-reacher-$(date +%Y-%m-%d)" \
    "run.exp_name=jaxgcrl-ppo-reacher-seed0-$(date +%Y-%m-%d)"