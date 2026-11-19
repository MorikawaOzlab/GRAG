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
    env=ant_u_maze \
    system.actor_lr=1.7e-05 \
    system.q_lr=3.9e-05 \
    system.clip_eps=0.16 \
    system.ent_coef=0.00036 \
    system.max_grad_norm=0.5 \
    system.energy_fn=l2 \
    system.terminate_on_success=false \
    system.log_std_max=0.575 \
    run.log_wandb=true \
    run.wandb_project_name=cppo-jaxgcrl \
    "run.wandb_group=jaxgcrl-cppo-ant_u_maze-$(date +%Y-%m-%d)" \
    "run.exp_name=jaxgcrl-cppo-ant_u_maze-seed0-$(date +%Y-%m-%d)"