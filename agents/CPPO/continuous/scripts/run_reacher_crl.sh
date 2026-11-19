#!/usr/bin/env bash
# CSAC in continuous_params.csv — maps to the CRL system in this codebase.
set -euo pipefail
source .venv/bin/activate

export WANDB_API_KEY=${WANDB_API_KEY:-}
export MUJOCO_GL=${MUJOCO_GL:-off}

if [ -f scripts/fix_execstack.py ]; then
    python scripts/fix_execstack.py >/dev/null 2>&1 || true
fi

python -u -m jaxgcrl.run \
    --config-name=crl \
    env=reacher \
    system.policy_lr=0.0003 \
    system.contrastive_loss_fn=bwd_infonce \
    system.energy_fn=norm \
    system.use_ln=false \
    run.log_wandb=true \
    run.wandb_project_name=cppo-jaxgcrl \
    "run.wandb_group=jaxgcrl-crl-reacher-$(date +%Y-%m-%d)" \
    "run.exp_name=jaxgcrl-crl-reacher-seed0-$(date +%Y-%m-%d)"