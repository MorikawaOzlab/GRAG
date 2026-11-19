#!/usr/bin/env bash
# SMAX 5m_vs_6m — CPPO, IPPO, ICSAC
set -euo pipefail
source .venv/bin/activate
export WANDB_API_KEY=${WANDB_API_KEY:-}
DATE=$(date +%Y-%m-%d)
SEEDS=${SEEDS:-0}
PROJECT=${PROJECT:-cppo}

CPPO=mava.systems.icrl.anakin.ff_ppo_crl
IPPO=mava.systems.ppo.anakin.ff_ippo
ICSAC=mava.systems.icrl.anakin.ff_icrl

COMMON="\
  env=smax \
  arch.num_envs=512 \
  system.num_updates=1250 \
  system.rollout_length=128 \
  arch.num_evaluation=80 \
  arch.num_eval_episodes=2048 \
  arch.absolute_metric=true \
  logger.loggers.wandb.enabled=true \
  logger.loggers.wandb.project=${PROJECT}"

python -u -m $CPPO -m system.seed=${SEEDS} \
  env.scenario.task_name=5m_vs_6m \
  system.gamma=0.99 system.clip_eps=0.01 system.ppo_epochs=2 \
  system.actor_lr=0.00078 system.q_lr=0.00087 \
  system.lr_decay_type=linear system.lr_end=1e-06 \
  system.ent_coef=0.01 system.max_grad_norm=10.0 \
  system.energy_fn=l2 system.logsumexp_penalty_coeff=0.1 \
  $COMMON \
  logger.loggers.wandb.group=paper-CPPO-smax-5m_vs_6m-${DATE} \
  logger.loggers.wandb.run_name=CPPO-smax-5m_vs_6m-l2

python -u -m $IPPO -m system.seed=${SEEDS} \
  env.scenario.task_name=5m_vs_6m \
  system.gamma=0.99 system.clip_eps=0.2 system.ppo_epochs=4 \
  system.num_minibatches=2 system.gae_lambda=0.95 system.vf_coef=0.5 \
  system.actor_lr=0.0005 system.critic_lr=0.0005 \
  system.ent_coef=0.01 system.max_grad_norm=0.5 \
  $COMMON \
  logger.loggers.wandb.group=paper-IPPO-smax-5m_vs_6m-${DATE} \
  logger.loggers.wandb.run_name=IPPO-smax-5m_vs_6m

python -u -m $ICSAC -m system.seed=${SEEDS} \
  env.scenario.task_name=5m_vs_6m \
  system.gamma=0.99 system.policy_lr=0.0003 system.q_lr=0.0003 system.alpha_lr=0.0003 \
  system.init_alpha=0.1 system.alpha_min=0.01 system.autotune=true \
  system.target_entropy_scale=0.5 system.buffer_size=5000 system.batch_size=256 \
  system.max_grad_norm=0.5 system.contrastive_loss_fn=fwd_infonce \
  system.energy_fn=norm system.logsumexp_penalty_coeff=0.1 \
  $COMMON \
  logger.loggers.wandb.group=paper-ICSAC-smax-5m_vs_6m-${DATE} \
  logger.loggers.wandb.run_name=ICSAC-smax-5m_vs_6m

echo "ALL DONE."
