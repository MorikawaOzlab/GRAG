#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export WANDB_API_KEY=${WANDB_API_KEY:-}

DATE=$(date +%Y-%m-%d)
SEEDS=${SEEDS:-0}                           
PROJECT=${PROJECT:-jaxnav}

CPPO=mava.systems.icrl.anakin.ppo_crl_jaxnav
IPPO=mava.systems.ppo.anakin.ff_ippo
ICSAC=mava.systems.icrl.anakin.ff_icrl_jaxnav

COMMON="\
  env=jaxnav \
  system.num_updates=1250 \
  arch.absolute_metric=true \
  logger.loggers.wandb.enabled=true \
  logger.loggers.wandb.project=${PROJECT}"



#  CPPO 
python -u -m $CPPO -m system.seed=${SEEDS} \
  system.gamma=0.9999 system.actor_lr=2.7041106512224436e-05 \
  system.clip_eps=0.11597317154664095 \
  system.logsumexp_penalty_coeff=0.298 \
  $COMMON \
  logger.loggers.wandb.group=cppo-jaxnav-${DATE} \
  logger.loggers.wandb.run_name=cppo-jaxnav

# IPPO (dense reward) 
python -u -m $IPPO -m system.seed=${SEEDS} \
  $COMMON \
  logger.loggers.wandb.group=ippo-jaxnav-${DATE} \
  logger.loggers.wandb.run_name=ippo-jaxnav

# ICSAC (off-policy contrastive) 
python -u -m $ICSAC -m system.seed=${SEEDS} \
  system.policy_lr=2e-05 system.q_lr=2e-05 system.alpha_lr=5e-06 \
  system.target_entropy_scale=3 system.add_agent_id=true \
  $COMMON \
  logger.loggers.wandb.group=icsac-jaxnav-${DATE} \
  logger.loggers.wandb.run_name=icsac-jaxnav

echo "ALL DONE."
