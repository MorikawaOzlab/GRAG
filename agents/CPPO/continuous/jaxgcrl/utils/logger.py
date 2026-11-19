"""Logger with console, W&B, and marl-eval JSON backends."""

import logging
import zipfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import numpy as np
from omegaconf import OmegaConf

try:
    from colorama import Fore, Style
except ImportError:  # graceful fallback if colorama isn't installed
    class _Dummy:
        def __getattr__(self, _):
            return ""
    Fore = Style = _Dummy()  # type: ignore


class LogEvent(Enum):
    ACT = "actor"
    TRAIN = "trainer"
    EVAL = "evaluator"
    ABSOLUTE = "absolute"
    MISC = "misc"


_EVENT_COLOURS = {
    LogEvent.TRAIN: Fore.MAGENTA,
    LogEvent.EVAL: Fore.GREEN,
    LogEvent.ABSOLUTE: Fore.BLUE,
    LogEvent.ACT: Fore.CYAN,
    LogEvent.MISC: Fore.YELLOW,
}


def _flatten(d: Dict[str, Any], sep: str = "/", parent_key: str = "") -> Dict[str, Any]:
    """Flatten a nested dict with `sep` as the key separator."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(_flatten(v, sep=sep, parent_key=kk))
        else:
            out[kk] = v
    return out


def _to_scalar(v):
    if isinstance(v, (jax.Array, np.ndarray)):
        try:
            return v.item() if v.ndim == 0 else float(np.mean(np.asarray(v)))
        except Exception:
            return float(np.asarray(v).flatten()[0])
    return v


_CONSOLE_EVAL_KEYS = {
    "episode_reward",
    "episode_success",
    "episode_success_any",
    "episode_success_easy",
    "avg_episode_length",
    "sps",
    "epoch_eval_time",
}


class _ConsoleLogger:
    """One coloured line per `log_dict` call, formatted `EVENT - k1: v1 | k2: v2 | …`."""

    def __init__(self) -> None:
        self.logger = logging.getLogger()

        # Strip Hydra's StreamHandler to avoid double-printing; keep FileHandlers.
        kept = []
        for h in self.logger.handlers:
            if isinstance(h, logging.FileHandler):
                kept.append(h)
            elif isinstance(h, logging.StreamHandler) and getattr(h, "_jaxgcrl_console", False):
                kept.append(h)  # already ours, idempotent on re-init
        self.logger.handlers = kept

        if not any(getattr(h, "_jaxgcrl_console", False) for h in self.logger.handlers):
            ch = logging.StreamHandler()
            ch._jaxgcrl_console = True  # type: ignore[attr-defined]
            ch.setFormatter(logging.Formatter(f"{Fore.CYAN}{Style.BRIGHT}%(message)s{Style.RESET_ALL}", "%H:%M:%S"))
            self.logger.addHandler(ch)
        self.logger.setLevel("INFO")

    def log_dict(self, data: Dict[str, Any], step: int, eval_step: int, event: LogEvent) -> None:
        flat = _flatten(data, sep=" ")
        colour = _EVENT_COLOURS[event]
        parts: List[str] = []
        for k, v in flat.items():
            if event == LogEvent.EVAL and k not in _CONSOLE_EVAL_KEYS:
                continue
            kk = k.replace("_", " ").capitalize()
            v = _to_scalar(v)
            parts.append(f"{kk}: {v:.3f}" if isinstance(v, float) else f"{kk}: {v}")
        if not parts:
            return
        msg = " | ".join(parts)
        self.logger.info(f"{colour}{Style.BRIGHT}{event.value.upper()} - step={step} | {msg}{Style.RESET_ALL}")

    def log_config(self, cfg: Dict[str, Any]) -> None:
        colour = _EVENT_COLOURS[LogEvent.MISC]
        self.logger.info(f"{colour}{Style.BRIGHT}CONFIG{Style.RESET_ALL}")
        try:
            from rich.pretty import pprint
            pprint(cfg)
        except ImportError:
            import pprint as _pp
            self.logger.info(_pp.pformat(cfg))

    def stop(self) -> None:
        return None


class _WandbLogger:
    """Wraps the standard `wandb.init / wandb.log` API. Reads run metadata
    from the hydra config via fields like `run.wandb_project_name`."""

    def __init__(self, cfg) -> None:
        import wandb  # local import — keeps the module importable on machines without wandb

        self._wandb = wandb
        run = cfg.get("run", {}) if hasattr(cfg, "get") else cfg.run
        wandb_cfg = cfg.logger.loggers.wandb

        self._upload_json_data: bool = bool(wandb_cfg.get("upload_json_data", False))
        self._json_path: Optional[Path] = None

        # Disabled mode for tests / no-network runs.
        mode = OmegaConf.to_container(run, resolve=True).get("wandb_mode", "online")  # type: ignore[arg-type]

        self.run = wandb.init(
            project=wandb_cfg.get("project", None) or run.get("wandb_project_name", "jaxgcrl"),
            entity=wandb_cfg.get("entity", None),
            group=wandb_cfg.get("group", None) or run.get("wandb_group", None),
            name=wandb_cfg.get("run_name", None) or run.get("exp_name", None),
            tags=list(wandb_cfg.get("tags", []) or []),
            mode=mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def set_json_path(self, path: Path) -> None:
        self._json_path = path

    def log_dict(self, data: Dict[str, Any], step: int, eval_step: int, event: LogEvent) -> None:
        flat = _flatten(data, sep="/")
        payload = {f"{event.value}/{k}": _to_scalar(v) for k, v in flat.items()}
        self._wandb.log(payload, step=int(step))

    def log_config(self, _cfg: Dict[str, Any]) -> None:
        return None

    def _zip_and_upload_json(self) -> None:
        if self._json_path is None or not self._json_path.exists():
            logging.warning("upload_json_data=True but metrics.json not found at %s", self._json_path)
            return
        zip_path = self._json_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(self._json_path, arcname=self._json_path.name)
        self.run.save(str(zip_path), policy="now")
        logging.info("Uploaded %s to WandB", zip_path.name)

    def stop(self) -> None:
        try:
            if self._upload_json_data:
                self._zip_and_upload_json()
            self.run.finish()
        except Exception:
            pass


class _JsonLogger:
    """marl-eval-compatible JSON writer (env→task→algo→seed_N→step_M tree).

    Wraps `marl_eval.json_tools.JsonLogger`. Writes ONLY on `LogEvent.EVAL`
    and `LogEvent.ABSOLUTE`. Filters incoming metrics through a key allowlist
    (with optional rename) defined in YAML — usually mapping JaxGCRL's
    `episode_reward` → marl-eval's `mean_episode_return` so plotters work.
    """

    def __init__(self, cfg) -> None:
        from marl_eval.json_tools import JsonLogger as _MELJsonLogger

        json_cfg = cfg.logger.loggers.json
        run = cfg.run

        path = json_cfg.get("path", None) or f"results/json/{cfg.env.env_name}"
        task_name = json_cfg.get("task_name", None) or cfg.env.env_name
        env_name = json_cfg.get("env_name", None) or cfg.env.env_name
        algo = cfg.get("agent", "agent")
        seed = int(run.seed)

        self._logger = _MELJsonLogger(
            path=path,
            algorithm_name=algo,
            task_name=task_name,
            environment_name=env_name,
            seed=seed,
        )

        # Allowlist: dict mapping incoming key -> json-output key (or None to keep).
        # Defaults match marl-eval plotting conventions.
        default_map = {
            "episode_reward": "mean_episode_return",
            "episode_success_any": "mean_episode_success",
            "sps": "steps_per_second",
        }
        user_map = json_cfg.get("metrics_to_log", None)
        self._key_map: Dict[str, str] = (
            dict(user_map) if user_map else default_map
        )

    @property
    def json_file_path(self) -> Path:
        return Path(self._logger.file_path)

    def log_dict(self, data: Dict[str, Any], step: int, eval_step: int, event: LogEvent) -> None:
        if event not in (LogEvent.EVAL, LogEvent.ABSOLUTE):
            return
        is_absolute = event == LogEvent.ABSOLUTE
        for k_in, k_out in self._key_map.items():
            if k_in not in data:
                continue
            v = _to_scalar(data[k_in])
            if not isinstance(v, (int, float)):
                continue
            self._logger.write(int(step), k_out, float(v), eval_step, is_absolute)

    def log_config(self, _cfg: Dict[str, Any]) -> None:
        return None

    def stop(self) -> None:
        return None  # marl_eval.JsonLogger flushes on every write


class MavaLogger:
    """Fans out to enabled logging backends (console, W&B, JSON)."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.backends: List[Any] = []

        loggers_cfg = cfg.logger.loggers if hasattr(cfg, "logger") else {}
        if loggers_cfg.console.get("enabled", True):
            self.backends.append(_ConsoleLogger())

        wandb_backend: Optional[_WandbLogger] = None
        if loggers_cfg.wandb.get("enabled", False):
            try:
                wandb_backend = _WandbLogger(cfg)
                self.backends.append(wandb_backend)
            except Exception as e:
                logging.warning("WandbLogger disabled: %s", e)

        json_backend: Optional[_JsonLogger] = None
        if loggers_cfg.json.get("enabled", False):
            try:
                json_backend = _JsonLogger(cfg)
                self.backends.append(json_backend)
            except Exception as e:
                logging.warning("JsonLogger disabled: %s", e)

        # Cross-wire: let WandB backend know where the json file lives so it
        # can upload it at stop() time when upload_json_data=True.
        if wandb_backend is not None and json_backend is not None:
            wandb_backend.set_json_path(json_backend.json_file_path)

    def log_config(self, cfg_dict: Dict[str, Any]) -> None:
        for b in self.backends:
            b.log_config(cfg_dict)

    def log(self, metrics: Dict[str, Any], t: int, t_eval: int, event: LogEvent) -> None:
        # Drop non-numeric / unprocessable fields silently; reduce arrays to scalar.
        clean = {}
        for k, v in metrics.items():
            try:
                clean[k] = _to_scalar(v)
            except Exception:
                continue
        for b in self.backends:
            b.log_dict(clean, t, t_eval, event)

    def stop(self) -> None:
        for b in self.backends:
            try:
                b.stop()
            except Exception:
                pass
