"""PyTorch port of CPPO's discrete Navix learner for the GridWorld runtime.

Reference: CPPO/mava/systems/icrl/anakin/ff_ppo_crl_nx.py.
GridWorld supplies one live environment. Collect sequential fixed-length
windows with a frozen policy in place of Navix's parallel streams.
"""

import base64
from copy import deepcopy
import io
import math
import secrets


ACTIONS = ("up", "down", "left", "right")
DEFAULTS = {
    "hidden_size": 1024, "depth": 4, "actor_hidden_size": 256, "repr_dim": 64,
    "rollout_steps": 128 * 512, "rollout_length": 128, "batch_size": 512,
    "epochs": 1, "total_updates": 1250, "learning_rate": 2.5e-4,
    "critic_learning_rate": 2.5e-4,
    "gamma": 0.99, "clip_coef": 0.2, "entropy_coef": 0.01,
    "max_grad_norm": 0.5, "lr_end": 1e-7, "logsumexp_penalty_coef": 0.1,
    "normalize_advantage": True, "evaluation_greedy": False,
    "energy_fn": "norm", "contrastive_loss_fn": "fwd_infonce", "lr_decay_type": "cosine",
}

CHOICES = {
    "energy_fn": ("norm", "l2", "dot", "cosine"),
    "contrastive_loss_fn": ("fwd_infonce", "bwd_infonce", "sym_infonce", "binary_nce"),
    "lr_decay_type": ("none", "linear", "cosine"),
}


class CPPOAgent:
    display_name = "CPPO"
    CHECKPOINT_INCLUDES_MODEL = True
    SCHEMA_VERSION = 3

    def __init__(self, settings):
        import torch

        self.torch = torch
        self.settings = deepcopy(settings)
        self.config = {}
        for key, default in DEFAULTS.items():
            if key == "critic_learning_rate":
                default = self.config["learning_rate"]
            value = settings.get("cppo_" + key, default)
            if key in CHOICES:
                valid = isinstance(value, str) and value in CHOICES[key]
            elif type(default) is bool:
                valid = type(value) is bool
            elif type(default) is int:
                minimum = 2 if key in {"rollout_length", "rollout_steps", "batch_size"} else 1
                valid = type(value) is int and minimum <= value <= 1_000_000
            else:
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            if not valid:
                raise ValueError(f"Invalid CPPO setting: cppo_{key}={value!r}")
            self.config[key] = value
        for key in ("gamma", "clip_coef"):
            if not 0 < self.config[key] < 1:
                raise ValueError(f"cppo_{key} must be between zero and one")
        min_lr = min(self.config["learning_rate"], self.config["critic_learning_rate"])
        if min_lr <= 0 or self.config["max_grad_norm"] <= 0:
            raise ValueError("CPPO learning rate and gradient norm limit must be positive")
        if self.config["lr_end"] > min_lr:
            raise ValueError("cppo_lr_end must not exceed either initial learning rate")

        self.settings.update({"cppo_" + key: value for key, value in self.config.items()})
        self.seed = settings.get("seed")
        if self.seed is None:
            self.seed = secrets.randbits(32)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.map = self.networks = self.actor_optimizer = self.critic_optimizer = None
        self.rollout = []
        self.pending = None
        self.episode_id = 0
        self.updates = self.gradient_steps = 0
        self.metrics = {}

    def _initialize(self, observation):
        layout = {key: deepcopy(observation[key]) for key in ("rows", "cols", "obstacles", "goal")}
        if self.map is not None:
            if layout != self.map:
                raise ValueError("CPPO requires a fixed GridWorld map within a session")
            return

        torch = self.torch
        nn = torch.nn
        rows, cols = layout["rows"], layout["cols"]
        # Navix uses a limited view. With no heading in GridWorld, use a
        # north-up neighborhood with radius three on large map dimensions.
        self.view_shape = (min(rows, 7), min(cols, 7))
        self.view_padding = (3 if rows > 7 else 0, 3 if cols > 7 else 0)
        features = math.prod(self.view_shape) * 64

        # Flax SAME padding puts the extra row and column after the input.
        def trunk():
            layers = []
            for inputs, outputs in ((3, 16), (16, 32), (32, 64)):
                layers.extend((nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(inputs, outputs, 2), nn.SiLU()))
            return nn.Sequential(*layers)

        def head(inputs, width, depth, outputs, layer_norm=False):
            layers = []
            for _ in range(depth):
                layers.append(nn.Linear(inputs, width))
                if layer_norm:
                    layers.append(nn.LayerNorm(width, eps=1e-6))
                layers.append(nn.SiLU())
                inputs = width
            return nn.Sequential(*layers, nn.Linear(inputs, outputs))

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            width, depth, rep = (self.config[key] for key in ("hidden_size", "depth", "repr_dim"))
            self.networks = nn.ModuleDict({
                "actor_trunk": trunk(),
                "actor": head(features + 4, self.config["actor_hidden_size"], 2, len(ACTIONS)),
                "sa_trunk": trunk(),
                "sa": head(features + 2 + len(ACTIONS), width, depth, rep, layer_norm=True),
                "goal": head(2, width, depth, rep, layer_norm=True),
            })
            # PyTorch's default Conv/Linear kernels already use the same
            # fan-in uniform bounds as variance_scaling(1/3, "fan_in", "uniform").
            for module in self.networks.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    nn.init.zeros_(module.bias)
            nn.init.orthogonal_(self.networks["actor"][-1].weight, 0.01)

        lr = self.config["learning_rate"]
        self.actor_optimizer = torch.optim.Adam(self._parameters("actor", "actor_trunk"), lr=lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self._parameters("sa", "sa_trunk", "goal"),
                                                 lr=self.config["critic_learning_rate"], eps=1e-8)
        self.map = layout
        self.target = torch.tensor(layout["goal"], dtype=torch.float32)
        self.grid = torch.zeros(3, rows, cols)
        for row, col in layout["obstacles"]:
            self.grid[0, row, col] = 1
        goal_row, goal_col = layout["goal"]
        self.grid[1, goal_row, goal_col] = 1
        pad_y, pad_x = self.view_padding
        if pad_y or pad_x:
            pad = (pad_x, pad_x, pad_y, pad_y)
            functional = torch.nn.functional
            self.grid = torch.cat((functional.pad(self.grid[:1], pad, value=1),
                                   functional.pad(self.grid[1:], pad)), dim=0)

    def _parameters(self, *names):
        return [param for name in names for param in self.networks[name].parameters()]

    def _state(self, observation):
        self._initialize(observation)
        return list(observation["position"])

    def _features(self, positions, trunk):
        torch = self.torch
        rows, cols = positions.long().unbind(dim=1)
        height, width = self.view_shape
        pad_y, pad_x = self.view_padding
        row_start = rows if pad_y else torch.zeros_like(rows)
        col_start = cols if pad_x else torch.zeros_like(cols)
        y = row_start[:, None, None] + torch.arange(height)[None, :, None]
        x = col_start[:, None, None] + torch.arange(width)[None, None, :]
        grid = self.grid[:, y, x].permute(1, 0, 2, 3)
        grid[torch.arange(len(positions)), 2, rows + pad_y - row_start, cols + pad_x - col_start] = 1
        return self.networks[trunk](grid).permute(0, 2, 3, 1).flatten(1)

    def _distribution(self, positions):
        torch = self.torch
        features = self._features(positions, "actor_trunk")
        inputs = torch.cat((features, positions.float(), self.target.expand(len(positions), -1)), dim=-1)
        return torch.distributions.Categorical(logits=self.networks["actor"](inputs))

    def choose_action(self, observation, greedy=False):
        torch = self.torch
        position = self._state(observation)
        with torch.no_grad():
            policy = self._distribution(torch.tensor([position]))
            if greedy:
                return ACTIONS[policy.probs[0].argmax().item()]
            if self.pending is not None:
                raise RuntimeError("Call learn() before sampling the next action")
            action = torch.multinomial(policy.probs[0], 1, generator=self.generator).item()
            self.pending = (position, action, policy.logits[0, action].item())
        return ACTIONS[action]

    def action_probabilities(self, observation):
        with self.torch.no_grad():
            policy = self._distribution(self.torch.tensor([self._state(observation)]))
            return dict(zip(ACTIONS, policy.probs[0].tolist()))

    def evaluation_action(self, observation):
        torch = self.torch
        position = self._state(observation)
        with torch.no_grad():
            probs = self._distribution(torch.tensor([position])).probs[0]
            if self.config["evaluation_greedy"]:
                return ACTIONS[probs.argmax().item()]
            return ACTIONS[torch.multinomial(probs, 1, generator=self.generator).item()]

    def learn(self, observation, action, reward, next_observation, reached_goal):
        if self.pending is None or ACTIONS[self.pending[1]] != action:
            raise RuntimeError("Call learn() with the action just sampled by choose_action()")
        achieved = self._state(next_observation)
        self.rollout.append([*self.pending, achieved, self.episode_id])
        self.pending = None
        if len(self.rollout) == self.config["rollout_steps"]:
            self._update()
            self.rollout.clear()

    def end_episode(self):
        if self.pending is not None:
            raise RuntimeError("Call learn() before end_episode()")
        self.episode_id += 1

    def _hindsight_indices(self, episode_ids, window_ends):
        """Sample the reference distribution in linear space."""
        torch = self.torch
        rows = torch.arange(len(episode_ids))
        terminal = window_ends.clone()
        terminal[:-1] |= episode_ids[:-1] != episode_ids[1:]
        ends = torch.where(terminal, rows, len(rows)).flip(0).cummin(0).values.flip(0)
        future_count = ends - rows

        # Weight gamma**offset for strictly future rows, plus 1e-5 for self.
        # Invert the CDF of the truncated geometric distribution.
        gamma = self.config["gamma"]
        log_gamma = math.log(gamma)
        future_mass_factor = -torch.expm1(future_count * torch.tensor(log_gamma, dtype=torch.float64))
        future_mass = future_mass_factor * (gamma / (1 - gamma))
        draw = torch.rand(len(rows), dtype=torch.float64, generator=self.generator) * (future_mass + 1e-5)
        future_draw = (draw - 1e-5).clamp_min(0) / future_mass.clamp_min(torch.finfo(torch.float64).tiny)
        max_probability = 1 - torch.finfo(torch.float64).eps
        offset = torch.log1p(-(future_draw * future_mass_factor).clamp_max(max_probability)) / log_gamma
        offset = torch.where(draw < 1e-5, 0, offset.floor().long() + 1)
        return rows + offset.minimum(future_count)

    def _relabel(self, achieved, episode_ids):
        """Drop each window's last row and reproduce time-major flattening."""
        torch = self.torch
        rows = torch.arange(len(episode_ids))
        length = self.config["rollout_length"]
        ends = (rows + 1) % length == 0
        ends[-1] = True
        goals = self._hindsight_indices(episode_ids, ends)
        retained = rows[~ends]
        window_count = (len(rows) + length - 1) // length
        order = (retained % length * window_count + retained // length).argsort()
        retained = retained[order]
        return retained, achieved[goals[retained]]

    def _sa_features(self, positions):
        return self.torch.cat((self._features(positions, "sa_trunk"), positions.float()), dim=-1)

    def _unit_embedding(self, embedding):
        return embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-6)

    def _energy(self, sa, goal):
        name = self.config["energy_fn"]
        if name == "cosine":
            sa, goal = self._unit_embedding(sa), self._unit_embedding(goal)
        if name in ("dot", "cosine"):
            return (sa * goal).sum(dim=-1)
        squared_distance = (sa - goal).square().sum(dim=-1)
        return self._distance_energy(squared_distance)

    def _distance_energy(self, squared_distance):
        if self.config["energy_fn"] == "norm":
            return -(squared_distance + 1e-6).sqrt()
        return -squared_distance

    def _all_q(self, positions):
        torch = self.torch
        count = len(ACTIONS)
        features = self._sa_features(positions)
        encoded = self.networks["sa"](torch.cat((
            features[:, None, :].expand(-1, count, -1), torch.eye(count).expand(len(positions), -1, -1)
        ), dim=-1))
        return self._energy(encoded, self.networks["goal"](self.target))

    def _advantages(self, positions, actions):
        torch = self.torch
        with torch.no_grad():
            parts = []
            size = self.config["batch_size"]
            for batch, taken in zip(positions.split(size), actions.split(size)):
                q = self._all_q(batch)
                baseline = (self._distribution(batch).probs * q).sum(dim=-1)
                parts.append(q.gather(1, taken[:, None]).squeeze(1) - baseline)
            advantages = torch.cat(parts)
            # The reference normalizes once over the retained rollout,
            # before shuffling, then freezes the result across all epochs.
            if self.config["normalize_advantage"]:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            return advantages

    def _contrastive_logits(self, sa, goal):
        name = self.config["energy_fn"]
        if name == "cosine":
            sa, goal = self._unit_embedding(sa), self._unit_embedding(goal)
        if name in ("dot", "cosine"):
            return sa @ goal.T
        # Direct cdist avoids cancellation from squared norms when pairs are
        # very similar and avoids an explicit [batch, batch, repr] tensor.
        distance = self.torch.cdist(sa, goal, compute_mode="donot_use_mm_for_euclid_dist")
        return self._distance_energy(distance.square())

    def _contrastive_loss(self, logits):
        name = self.config["contrastive_loss_fn"]
        positives = logits.diag()
        if name == "binary_nce":
            torch = self.torch
            functional = torch.nn.functional
            diagonal = torch.eye(len(logits), dtype=torch.bool, device=logits.device)
            loss = functional.softplus(-positives).mean()
            loss = loss + functional.softplus(logits).masked_fill(diagonal, 0).mean()
        else:
            loss = -positives.mean() * (2 if name == "sym_infonce" else 1)
            if name in ("fwd_infonce", "sym_infonce"):
                loss = loss + logits.logsumexp(dim=1).mean()
            if name in ("bwd_infonce", "sym_infonce"):
                loss = loss + logits.logsumexp(dim=0).mean()
        coefficient = self.config["logsumexp_penalty_coef"]
        if coefficient:
            loss = loss + coefficient * (logits + 1e-6).logsumexp(dim=1).square().mean()
        return loss

    def _decayed_rate(self, start, schedule_steps):
        decay = self.config["lr_decay_type"]
        if decay == "none":
            return start
        progress = min(self.gradient_steps / schedule_steps, 1)
        multiplier = 1 - progress if decay == "linear" else (1 + math.cos(math.pi * progress)) / 2
        end = self.config["lr_end"]
        return end + (start - end) * multiplier

    def _contrastive_metrics(self, logits):
        torch = self.torch
        with torch.no_grad():
            positives = logits.diag()
            other_pairs = logits.numel() - len(positives)
            negatives = (logits.sum() - positives.sum()) / max(1, other_pairs)
            accuracy = (logits.argmax(dim=1) == torch.arange(len(positives))).float().mean()
            return positives.mean(), negatives, accuracy, (logits + 1e-6).logsumexp(dim=1).mean()

    def _policy_snapshot(self, positions):
        with self.torch.no_grad():
            return self.torch.cat([self._distribution(part).logits
                                   for part in positions.split(self.config["batch_size"])])

    def _update(self):
        torch = self.torch
        positions, actions, old_logs, achieved, ids = map(torch.tensor, zip(*self.rollout))
        indices, goals = self._relabel(achieved, ids)
        # Small GridWorld runs may request fewer usable rows than a Navix batch.
        size = min(len(indices), self.config["batch_size"])
        count = len(indices) // size * size
        indices, goals = indices[:count], goals[:count].float()
        positions, actions, old_logs = positions[indices], actions[indices], old_logs[indices]
        advantages = self._advantages(positions, actions)
        behavior_logs = self._policy_snapshot(positions)
        batches = count // size
        schedule_steps = self.config["total_updates"] * self.config["epochs"] * batches
        metric_names = ("contrastive_loss", "policy_loss", "entropy", "logits_pos", "logits_neg",
                        "categorical_accuracy", "logsumexp")
        batch_metrics = torch.zeros(len(metric_names))
        clip = self.config["clip_coef"]
        for _ in range(self.config["epochs"]):
            for batch in torch.randperm(count, generator=self.generator).split(size):
                actor_lr = self._decayed_rate(self.config["learning_rate"], schedule_steps)
                critic_lr = self._decayed_rate(self.config["critic_learning_rate"], schedule_steps)

                policy = self._distribution(positions[batch])
                ratio = (policy.log_prob(actions[batch]) - old_logs[batch]).exp()
                advantage = advantages[batch]
                actor_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
                entropy = policy.entropy().mean()

                one_hot = torch.nn.functional.one_hot(actions[batch], len(ACTIONS)).float()
                sa = self.networks["sa"](torch.cat((self._sa_features(positions[batch]), one_hot), dim=-1))
                logits = self._contrastive_logits(sa, self.networks["goal"](goals[batch]))
                critic_loss = self._contrastive_loss(logits)
                actor_objective = actor_loss - self.config["entropy_coef"] * entropy
                for optimizer, loss, lr in ((self.actor_optimizer, actor_objective, actor_lr),
                                            (self.critic_optimizer, critic_loss, critic_lr)):
                    group = optimizer.param_groups[0]
                    group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(group["params"], self.config["max_grad_norm"])
                    optimizer.step()
                self.gradient_steps += 1
                batch_metrics += torch.stack((critic_loss.detach(), actor_loss.detach(), entropy.detach(),
                                              *self._contrastive_metrics(logits)))
        self.updates += 1
        current_logs = self._policy_snapshot(positions)
        self.metrics = {
            "updates": self.updates, "rollout_steps": len(self.rollout), "learning_rate": actor_lr,
            "critic_learning_rate": critic_lr,
            "rollout_episodes": ids.unique().numel(), "training_samples": count,
            "unique_achieved_goals": achieved.unique(dim=0).shape[0],
            "target_seen_in_rollout": bool((achieved == self.target).all(dim=1).any()),
            "policy_entropy": float(-(current_logs.exp() * current_logs).sum(dim=1).mean()),
            "behavior_entropy": float(-(behavior_logs.exp() * behavior_logs).sum(dim=1).mean()),
            "policy_kl": float((behavior_logs.exp() * (behavior_logs - current_logs)).sum(dim=1).mean()),
            "advantage_std": float(advantages.std(unbiased=False)),
            **dict(zip(metric_names, (batch_metrics / (self.config["epochs"] * batches)).tolist())),
        }

    def _encode(self, value):
        stream = io.BytesIO()
        self.torch.save(value, stream)
        return base64.b64encode(stream.getvalue()).decode("ascii")

    def _decode(self, value):
        return self.torch.load(io.BytesIO(base64.b64decode(value)), map_location="cpu", weights_only=True)

    def export(self):
        return {
            "algorithm": "cppo", "schema_version": self.SCHEMA_VERSION,
            "settings": deepcopy(self.settings), "seed": self.seed, "map": deepcopy(self.map),
            "updates": self.updates, "gradient_steps": self.gradient_steps, "metrics": deepcopy(self.metrics),
            "networks": None if self.networks is None else self._encode(self.networks.state_dict()),
        }

    def checkpoint(self):
        if self.pending is not None:
            raise RuntimeError("Save the checkpoint after calling learn()")
        return {
            "model": self.export(), "generator": self.generator.get_state().tolist(),
            "actor_optimizer": None if self.actor_optimizer is None else self._encode(self.actor_optimizer.state_dict()),
            "critic_optimizer": None if self.critic_optimizer is None else self._encode(self.critic_optimizer.state_dict()),
            "episode_id": self.episode_id, "rollout": deepcopy(self.rollout),
        }

    def load_checkpoint(self, state):
        model = state["model"]
        if model.get("algorithm") != "cppo" or model.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("This CPPO checkpoint is incompatible with the Navix-style CNN and optimizers")
        restored = type(self)({**model["settings"], "seed": model["seed"]})
        if model["map"] is not None:
            restored._initialize(model["map"])
            restored.networks.load_state_dict(restored._decode(model["networks"]))
            restored.actor_optimizer.load_state_dict(restored._decode(state["actor_optimizer"]))
            restored.critic_optimizer.load_state_dict(restored._decode(state["critic_optimizer"]))
        restored.generator.set_state(self.torch.tensor(state["generator"], dtype=self.torch.uint8))
        restored.updates = model["updates"]
        restored.gradient_steps = model["gradient_steps"]
        restored.metrics = deepcopy(model["metrics"])
        restored.rollout = deepcopy(state["rollout"])
        restored.episode_id = state["episode_id"]
        self.__dict__.update(restored.__dict__)

    def evaluation_copy(self):
        clone = type(self)({**self.settings, "seed": self.seed})
        if self.map is not None:
            clone._initialize(self.map)
            clone.networks.load_state_dict(self.networks.state_dict())
        return clone
