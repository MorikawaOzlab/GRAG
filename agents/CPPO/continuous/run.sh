#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

export WANDB_API_KEY=${WANDB_API_KEY:-}
export MUJOCO_GL=${MUJOCO_GL:-off}

if [ -f scripts/fix_execstack.py ]; then
    python scripts/fix_execstack.py >/dev/null 2>&1 || true
fi


bash scripts/run_ant_cppo.sh
bash scripts/run_ant_ppo.sh
bash scripts/run_ant_crl.sh
bash scripts/run_ant_u_maze_ppo.sh
bash scripts/run_reacher_ppo.sh
bash scripts/run_ant_u_maze_cppo.sh
bash scripts/run_reacher_cppo.sh
bash scripts/run_ant_u_maze_crl.sh
bash scripts/run_reacher_crl.sh