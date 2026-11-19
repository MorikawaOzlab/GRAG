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
    env=ant \
    system.discounting=0.995 \
    system.num_updates_per_batch=4 \
    system.reward_scaling=1.0 \
    run.log_wandb=true \
    run.wandb_project_name=cppo-jaxgcrl \
    "run.wandb_group=jaxgcrl-ppo-ant-$(date +%Y-%m-%d)" \
    "run.exp_name=jaxgcrl-ppo-ant-seed0-$(date +%Y-%m-%d)"