#!/usr/bin/env bash
# Connector con-5x5x3a — CPPO, IPPO, ICSAC
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
  env=vector-connector \
  arch.num_envs=512 \
  system.num_updates=1250 \
  system.rollout_length=128 \
  arch.num_evaluation=80 \
  arch.num_eval_episodes=2048 \
  arch.absolute_metric=true \
  logger.loggers.wandb.enabled=false \
  logger.loggers.wandb.project=${PROJECT}"

# CPPO params are identical across all connector scenarios
CPPO_ARGS="\
  system.gamma=0.99 system.clip_eps=0.2 system.ppo_epochs=1 \
  system.actor_lr=0.00025 system.q_lr=0.00025 \
  system.lr_decay_type=cosine system.lr_end=1e-07 \
  system.ent_coef=0.1 system.max_grad_norm=0.5 \
  system.energy_fn=norm system.logsumexp_penalty_coeff=0.1"

python -u -m $CPPO -m system.seed=${SEEDS} \
  env/scenario=con-5x5x3a \
  $CPPO_ARGS \
  $COMMON \
  logger.loggers.wandb.group=paper-CPPO-connector-con-5x5x3a-${DATE} \
  logger.loggers.wandb.run_name=CPPO-connector-con-5x5x3a

python -u -m $IPPO -m system.seed=${SEEDS} \
  env/scenario=con-5x5x3a \
  system.gamma=0.99 system.clip_eps=0.2 system.ppo_epochs=4 \
  system.num_minibatches=2 system.gae_lambda=0.95 system.vf_coef=0.5 \
  system.actor_lr=0.00025 system.critic_lr=0.00025 \
  system.ent_coef=0.01 system.max_grad_norm=0.5 \
  $COMMON \
  logger.loggers.wandb.group=paper-IPPO-connector-con-5x5x3a-${DATE} \
  logger.loggers.wandb.run_name=IPPO-connector-con-5x5x3a

python -u -m $ICSAC -m system.seed=${SEEDS} \
  env/scenario=con-5x5x3a \
  system.gamma=0.99 system.policy_lr=0.0003 system.q_lr=0.0003 system.alpha_lr=0.0003 \
  system.init_alpha=0.1 system.alpha_min=0.01 system.autotune=true \
  system.target_entropy_scale=0.5 system.buffer_size=5000 system.batch_size=256 \
  system.max_grad_norm=0.5 system.contrastive_loss_fn=fwd_infonce \
  system.energy_fn=norm system.logsumexp_penalty_coeff=0.1 \
  $COMMON \
  logger.loggers.wandb.group=paper-ICSAC-connector-con-5x5x3a-${DATE} \
  logger.loggers.wandb.run_name=ICSAC-connector-con-5x5x3a

echo "ALL DONE."
