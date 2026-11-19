#!/usr/bin/env bash
# Navix-DoorKey-Random-16x16-v0 — CPPO, PPO, CDQN
set -euo pipefail
source .venv/bin/activate
export WANDB_API_KEY=${WANDB_API_KEY:-}
DATE=$(date +%Y-%m-%d)
SEEDS=${SEEDS:-0}
PROJECT=${PROJECT:-cppo}

CPPO_NX=mava.systems.icrl.anakin.ff_ppo_crl_nx
IPPO_NX=mava.systems.ppo.anakin.ff_ppo_nx
DQN_CRL_NX=mava.systems.icrl.anakin.dqn_crl_nx

COMMON="\
  arch.num_envs=512 \
  system.num_updates=1250 \
  system.rollout_length=128 \
  arch.num_evaluation=80 \
  arch.num_eval_episodes=2048 \
  arch.absolute_metric=true \
  logger.loggers.wandb.enabled=false \
  logger.loggers.wandb.project=${PROJECT}"

python -u -m $CPPO_NX -m system.seed=${SEEDS} \
  env.env_name=Navix-DoorKey-Random-16x16-v0 env.task_name=Navix-DoorKey-Random-16x16-v0 \
  system.gamma=0.99 system.clip_eps=0.2 system.ppo_epochs=1 \
  system.actor_lr=0.00025 system.q_lr=0.00025 \
  system.lr_decay_type=cosine system.lr_end=1e-07 \
  system.ent_coef=0.5 system.max_grad_norm=0.5 \
  system.energy_fn=norm system.logsumexp_penalty_coeff=0.1 \
  $COMMON \
  logger.loggers.wandb.group=paper-CPPO-nx-doorkey-random-16x16-${DATE} \
  logger.loggers.wandb.run_name=CPPO-nx-doorkey-random-16x16

python -u -m $IPPO_NX -m system.seed=${SEEDS} \
  env.env_name=Navix-DoorKey-Random-16x16-v0 env.task_name=Navix-DoorKey-Random-16x16-v0 \
  system.gamma=0.999 system.clip_eps=0.2 system.ppo_epochs=4 \
  system.num_minibatches=8 system.gae_lambda=0.981 system.vf_coef=0.75 \
  system.actor_lr=0.00042 system.critic_lr=0.0007 \
  system.lr_decay_type=cosine system.lr_end=3.3e-06 \
  system.ent_coef=0.01 system.max_grad_norm=0.5 \
  $COMMON \
  logger.loggers.wandb.group=paper-PPO-nx-doorkey-random-16x16-${DATE} \
  logger.loggers.wandb.run_name=PPO-nx-doorkey-random-16x16

python -u -m $DQN_CRL_NX -m system.seed=${SEEDS} \
  env.env_name=Navix-DoorKey-Random-16x16-v0 env.task_name=Navix-DoorKey-Random-16x16-v0 \
  system.gamma=0.999 system.q_lr=0.0003 system.max_grad_norm=0.5 \
  system.buffer_size=5000 \
  system.energy_fn=norm system.logsumexp_penalty_coeff=0.1 \
  system.eps_start=1 system.eps_decay=50000 system.eps_min=0.05 \
  system.explore_steps=1000 \
  $COMMON \
  logger.loggers.wandb.group=paper-CDQN-nx-doorkey-random-16x16-${DATE} \
  logger.loggers.wandb.run_name=CDQN-nx-doorkey-random-16x16

echo "ALL DONE."
