# CPPO
# Switching between discrete and continuous baselines

This repo is a monorepo with **two isolated subprojects** that don't share dependencies:

| Subproject | Path | Envs | Algos |
|---|---|---|---|
| **Discrete** (mava) | `./` (root) | SMAX, Vector-Connector, Navix, JaxNav\* | ff_ippo, ff_mappo, ff_ppo_crl, ff_icrl, ppo_crl_jaxnav, ff_icrl_jaxnav |
| **Continuous** (jaxgcrl) | `./continuous/` | Ant, reacher, ant_maze | ppo, cppo, crl |

\* **JaxNav** is a continuous-action *multi-agent* environment. It lives in the `mava` subproject (which hosts the shared multi-agent infrastructure) and is run through the `*_jaxnav` systems via `scripts/jaxnav.sh`.

Each subproject has its **own** `pyproject.toml`, `uv.lock`, and `.venv`.

---


## Discrete (root)

```bash
# install 
uv sync --extra cuda12
source .venv/bin/activate

# run baselines
bash scripts/connector_con_5x5x3a.sh

# JaxNav (continuous-action MARL) — CPPO, IPPO, ICSAC
bash scripts/jaxnav.sh

```

---

## Continuous (jaxgcrl)

```bash
# install — separate venv inside continuous/
cd continuous/
uv sync
source .venv/bin/activate

# run baselines
bash scripts/run_ant_cppo.sh
```


## Hyperparameters
A separate CSV containing all hyperparameters used in our experiments will be released alongside the paper. These optimal hyperparameters are set by default in each runner script, overriding the Hydra defaults with env/system-specific values.
