#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

export WANDB_API_KEY=${WANDB_API_KEY:-}
export MUJOCO_GL=${MUJOCO_GL:-off}

if [ -f scripts/fix_execstack.py ]; then
    python scripts/fix_execstack.py >/dev/null 2>&1 || true
fi

python -u -m jaxgcrl.run \
    --config-name=cppo \
    env=ant \
    system.actor_lr=0.0001 \
    system.q_lr=8.5e-05 \
    system.ent_coef=0.00047 \
    system.lr_linear_decay=true \
    system.lr_end=1e-06 \
    system.log_std_max=1.5 \
    system.terminate_on_success=false \
    run.log_wandb=true \
    run.wandb_project_name=cppo-jaxgcrl \
    "run.wandb_group=jaxgcrl-cppo-ant-$(date +%Y-%m-%d)" \
    "run.exp_name=jaxgcrl-cppo-ant-seed0-$(date +%Y-%m-%d)"

