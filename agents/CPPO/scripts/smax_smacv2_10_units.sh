#!/usr/bin/env bash
# SMAX smacv2_10_units — CPPO, IPPO, ICSAC
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
  logger.loggers.wandb.enabled=false \
  logger.loggers.wandb.project=${PROJECT}"

python -u -m $CPPO -m system.seed=${SEEDS} \
  env.scenario.task_name=smacv2_10_units \
  system.gamma=0.99 system.clip_eps=0.2 system.ppo_epochs=1 \
  system.actor_lr=0.0006654299226364461 system.q_lr=0.00019825901799915864 \
  system.lr_decay_type=cosine system.lr_end=1e-06 \
  system.ent_coef=0.05 system.max_grad_norm=1 \
  system.energy_fn=norm system.logsumexp_penalty_coeff=0.4516040120578467 \
  $COMMON \
  logger.loggers.wandb.group=paper-CPPO-smax-smacv2_10_units-${DATE} \
  logger.loggers.wandb.run_name=CPPO-smax-smacv2_10_units

python -u -m $IPPO -m system.seed=${SEEDS} \
  env.scenario.task_name=smacv2_10_units \
  system.gamma=0.99 system.clip_eps=0.05 system.ppo_epochs=8 \
  system.num_minibatches=8 system.gae_lambda=0.95 system.vf_coef=0.5 \
  system.actor_lr=0.00025 system.critic_lr=0.0005 \
  system.ent_coef=0 system.max_grad_norm=5 \
  $COMMON \
  logger.loggers.wandb.group=paper-IPPO-smax-smacv2_10_units-${DATE} \
  logger.loggers.wandb.run_name=IPPO-smax-smacv2_10_units

# ICSAC used rollout_length=256, num_envs=256 for smacv2_10_units
python -u -m $ICSAC -m system.seed=${SEEDS} \
  env.scenario.task_name=smacv2_10_units \
  system.gamma=0.99 system.policy_lr=0.0003 system.q_lr=0.0003 system.alpha_lr=0.0003 \
  system.init_alpha=0.1 system.alpha_min=0.01 system.autotune=true \
  system.target_entropy_scale=0.5 system.buffer_size=5000 system.batch_size=256 \
  system.max_grad_norm=0.5 system.contrastive_loss_fn=fwd_infonce \
  system.energy_fn=norm system.logsumexp_penalty_coeff=0.1 \
  system.rollout_length=256 arch.num_envs=256 \
  $COMMON \
  logger.loggers.wandb.group=paper-ICSAC-smax-smacv2_10_units-${DATE} \
  logger.loggers.wandb.run_name=ICSAC-smax-smacv2_10_units

echo "ALL DONE."
