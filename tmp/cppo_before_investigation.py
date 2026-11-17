"""Single-file GridWorld adaptation of Contrastive PPO (arXiv:2605.13554v1).

Implements Eqs. 3, 6, 7, 9 and Algorithm 1 without environment rewards,
replay, target networks or a value head. The UI supplies full map observations;
the achieved/target goal is the grid coordinate, as in the paper's Navix tasks.
Unlike the paper's 512 parallel environments, complete episodes are collected
sequentially under one frozen policy, then discarded after an on-policy update.
Defaults follow Appendix D where applicable. HER uses a truncated discounted
future distribution; same-episode negatives are masked. Advantages are frozen
before encoder updates following Algorithm 1. See GridWorld/CPPO.md for details.

Only dependency: torch. No imports from other agents or the GridWorld runtime.
"""

from copy import deepcopy
import math
import secrets


ACTIONS = ("up", "down", "left", "right")


class CPPOAgent:
    display_name = "CPPO"

    def __init__(self, settings: dict):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("CPPOにはPyTorchが必要です。python -m pip install torch を実行してください。") from error
        self.torch = torch
        self.settings = deepcopy(settings)
        self.config = {}
        defaults = {"hidden_size": 512, "depth": 4, "repr_dim": 64,
                    "rollout_steps": 128, "batch_size": 256, "epochs": 1,
                    "total_updates": 1250, "learning_rate": 2.5e-4,
                    "gamma": .99, "clip_coef": .2, "entropy_coef": .01,
                    "max_grad_norm": .5, "lr_end": 1e-7}
        integer_keys = {"hidden_size", "depth", "repr_dim", "rollout_steps",
                        "batch_size", "epochs", "total_updates"}
        for key, default in defaults.items():
            value = settings.get("cppo_" + key, default)
            if key in integer_keys:
                minimum = 2 if key in {"batch_size", "rollout_steps"} else 1
                if type(value) is not int or not minimum <= value <= 1_000_000:
                    raise ValueError(f"cppo_{key}は{minimum}以上の整数にしてください。")
            elif (isinstance(value, bool) or not isinstance(value, (int, float))
                  or not math.isfinite(value) or value < 0):
                raise ValueError(f"cppo_{key}は有限の非負数にしてください。")
            self.config[key] = value
        if not 0 < self.config["gamma"] < 1:
            raise ValueError("cppo_gammaは0より大きく1未満にしてください。")
        if not 0 < self.config["clip_coef"] < 1:
            raise ValueError("cppo_clip_coefは0より大きく1未満にしてください。")
        if self.config["learning_rate"] <= 0 or self.config["max_grad_norm"] <= 0:
            raise ValueError("CPPOの学習率と勾配上限は正数にしてください。")
        if self.config["lr_end"] > self.config["learning_rate"]:
            raise ValueError("cppo_lr_endは初期学習率以下にしてください。")
        self.settings.update({"cppo_" + key: value for key, value in self.config.items()})
        self.seed = settings.get("seed")
        if self.seed is None:
            self.seed = secrets.randbits(32)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.map = None
        self.networks = self.actor = self.sa_encoder = self.goal_encoder = None
        self.optimizer = None
        self.episodes = []
        self.trajectory = []
        self.pending = None
        self.updates = 0
        self.metrics = {}

    def _initialize(self, observation):
        torch = self.torch
        nn = torch.nn
        layout = {key: deepcopy(observation[key]) for key in ("rows", "cols", "obstacles", "goal")}
        if self.map is not None:
            if layout != self.map:
                raise ValueError("CPPOの学習中にマップを変更できません。")
            return
        self.map = layout
        cells = layout["rows"] * layout["cols"]

        def mlp(inputs, outputs, output_gain=1.0):
            layers = []
            for _ in range(self.config["depth"]):
                layer = nn.Linear(inputs, self.config["hidden_size"])
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
                layers.extend([layer, nn.LayerNorm(self.config["hidden_size"]), nn.SiLU()])
                inputs = self.config["hidden_size"]
            layer = nn.Linear(inputs, outputs)
            nn.init.orthogonal_(layer.weight, output_gain)
            nn.init.zeros_(layer.bias)
            return nn.Sequential(*layers, layer)

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.actor = mlp(2 * cells + 2, 4, .01)
            self.sa_encoder = mlp(2 * cells + 4, self.config["repr_dim"])
            self.goal_encoder = mlp(2, self.config["repr_dim"])
            self.networks = nn.ModuleDict({"actor": self.actor, "sa": self.sa_encoder,
                                          "goal": self.goal_encoder})
        self.optimizer = torch.optim.Adam(self.networks.parameters(),
                                          lr=self.config["learning_rate"], eps=1e-5)

    def _state(self, observation):
        self._initialize(observation)
        cells = self.map["rows"] * self.map["cols"]
        state = [0.] * (2 * cells)
        for row, col in self.map["obstacles"]:
            state[row * self.map["cols"] + col] = 1.
        row, col = observation["position"]
        state[cells + row * self.map["cols"] + col] = 1.
        return state

    def _distribution(self, states):
        target = self.torch.tensor(self.map["goal"], dtype=self.torch.float32)
        goals = target.expand(states.shape[0], -1)
        return self.torch.distributions.Categorical(logits=self.actor(self.torch.cat([states, goals], -1)))

    def choose_action(self, observation, greedy=False):
        torch = self.torch
        state = self._state(observation)
        with torch.no_grad():
            distribution = self._distribution(torch.tensor([state]))
            if greedy:
                return ACTIONS[int(distribution.probs.argmax(-1).item())]
            if self.pending is not None:
                raise RuntimeError("次の行動選択前にlearn()を呼んでください。")
            action = int(torch.multinomial(distribution.probs[0], 1, generator=self.generator).item())
            self.pending = {"state": state, "action": action,
                            "log_prob": float(distribution.log_prob(torch.tensor(action)).item())}
        return ACTIONS[action]

    def action_probabilities(self, observation):
        torch = self.torch
        with torch.no_grad():
            probabilities = self._distribution(torch.tensor([self._state(observation)])).probs[0].tolist()
        return dict(zip(ACTIONS, probabilities))

    def learn(self, observation, action, reward, next_observation, reached_goal):
        # reward is deliberately unused: success is reported by the runtime only.
        if self.pending is None or ACTIONS[self.pending["action"]] != action:
            raise RuntimeError("learn()には直前に選択した行動を渡してください。")
        self._initialize(next_observation)
        self.trajectory.append({**self.pending, "achieved": list(next_observation["position"])})
        self.pending = None

    def set_episode_limit(self, max_steps):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_stepsは1以上の整数にしてください。")
        self.settings["max_steps"] = max_steps

    def end_episode(self):
        if self.pending is not None:
            raise RuntimeError("learn()の後にend_episode()を呼んでください。")
        if self.trajectory:
            self.episodes.append(self.trajectory)
            self.trajectory = []
        if (len(self.episodes) >= 2
                and sum(map(len, self.episodes)) >= self.config["rollout_steps"]):
            self._update()
            self.episodes.clear()

    def _hindsight(self):
        """Future achieved positions, never crossing terminal/time-limit boundaries."""
        torch = self.torch
        transitions, goals, episode_ids = [], [], []
        for episode_id, episode in enumerate(self.episodes):
            for time, transition in enumerate(episode):
                weights = self.config["gamma"] ** torch.arange(len(episode) - time)
                offset = int(torch.multinomial(weights, 1, generator=self.generator).item())
                transitions.append(transition)
                goals.append(episode[time + offset]["achieved"])
                episode_ids.append(episode_id)
        return transitions, torch.tensor(goals, dtype=torch.float32), torch.tensor(episode_ids)

    def _all_q(self, states):
        torch = self.torch
        count = len(states)
        actions = torch.eye(4).repeat(count, 1)
        embeddings = self.sa_encoder(torch.cat([states.repeat_interleave(4, 0), actions], -1))
        target = self.goal_encoder(torch.tensor([self.map["goal"]], dtype=torch.float32))
        return -torch.linalg.vector_norm(embeddings - target, dim=-1).reshape(count, 4)

    def _advantages(self, states, actions):
        # Exact discrete baseline, no sampled approximation, GAE or reward.
        with self.torch.no_grad():
            q = self._all_q(states)
            baseline = (self._distribution(states).probs * q).sum(-1)
            return q.gather(1, actions[:, None]).squeeze(1) - baseline

    def _update(self):
        torch = self.torch
        functional = torch.nn.functional
        rows, goals, episode_ids = self._hindsight()
        states = torch.tensor([row["state"] for row in rows], dtype=torch.float32)
        actions = torch.tensor([row["action"] for row in rows])
        old_logs = torch.tensor([row["log_prob"] for row in rows])
        advantages = self._advantages(states, actions)
        progress = min(self.updates / self.config["total_updates"], 1.)
        lr = self.config["lr_end"] + .5 * (self.config["learning_rate"] - self.config["lr_end"]) * (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        losses = []
        for _ in range(self.config["epochs"]):
            permutation = torch.randperm(len(rows), generator=self.generator)
            for indices in permutation.split(self.config["batch_size"]):
                # Bound the contrastive matrix by minibatch size. A homogeneous
                # minibatch gets one additional negative from another episode.
                candidates = indices
                if len(episode_ids[indices].unique()) == 1:
                    other = torch.nonzero(episode_ids != episode_ids[indices[0]]).flatten()
                    pick = torch.randint(len(other), (1,), generator=self.generator)
                    candidates = torch.cat([indices, other[pick]])
                phi = self.sa_encoder(torch.cat([states[indices], functional.one_hot(actions[indices], 4).float()], -1))
                psi = self.goal_encoder(goals[candidates])
                logits = -torch.linalg.vector_norm(phi[:, None, :] - psi[None, :, :], dim=-1)
                labels = torch.arange(len(indices))
                allowed = episode_ids[indices, None] != episode_ids[candidates][None, :]
                allowed[labels, labels] = True
                contrastive = functional.cross_entropy(logits.masked_fill(~allowed, -torch.inf), labels)
                distribution = self._distribution(states[indices])
                ratio = (distribution.log_prob(actions[indices]) - old_logs[indices]).exp()
                advantage = advantages[indices]
                if len(indices) > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
                clipped = ratio.clamp(1 - self.config["clip_coef"], 1 + self.config["clip_coef"])
                policy = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                entropy = distribution.entropy().mean()
                loss = contrastive + policy - self.config["entropy_coef"] * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.networks.parameters(), self.config["max_grad_norm"])
                self.optimizer.step()
                losses.append((float(contrastive.detach()), float(policy.detach()), float(entropy.detach())))
        self.updates += 1
        self.metrics = {"updates": self.updates, "rollout_steps": len(rows), "learning_rate": lr,
                        **{name: sum(row[i] for row in losses) / len(losses)
                           for i, name in enumerate(("contrastive_loss", "policy_loss", "entropy"))}}

    def _encode(self, value):
        if isinstance(value, self.torch.Tensor):
            return {"dtype": str(value.dtype).split(".")[-1], "values": value.detach().cpu().tolist()}
        if isinstance(value, dict):
            return {key: self._encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._encode(item) for item in value]
        return value

    def _decode(self, value):
        if isinstance(value, dict):
            if set(value) == {"dtype", "values"}:
                return self.torch.tensor(value["values"], dtype=getattr(self.torch, value["dtype"]))
            return {key: self._decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._decode(item) for item in value]
        return value

    def export(self):
        return {"algorithm": "cppo", "schema_version": 1, "settings": deepcopy(self.settings),
                "seed": self.seed, "map": deepcopy(self.map), "updates": self.updates,
                "metrics": deepcopy(self.metrics),
                "networks": None if self.networks is None else self._encode(self.networks.state_dict())}

    def checkpoint(self):
        if self.pending is not None:
            raise RuntimeError("learn()の後で保存してください。")
        return {"model": self.export(), "generator": self.generator.get_state().tolist(),
                "optimizer": None if self.optimizer is None else self._encode(self.optimizer.state_dict()),
                "episodes": deepcopy(self.episodes), "trajectory": deepcopy(self.trajectory)}

    def load_checkpoint(self, state):
        model = state["model"]
        if model.get("algorithm") != "cppo" or model.get("schema_version") != 1:
            raise ValueError("対応していないCPPOの保存形式です。")
        restored = type(self)({**model["settings"], "seed": model["seed"]})
        if model["map"] is not None:
            restored._initialize(model["map"])
            restored.networks.load_state_dict(restored._decode(model["networks"]))
            if state.get("optimizer") is not None:
                optimizer = restored._decode(state["optimizer"])
                optimizer["state"] = {int(key): value for key, value in optimizer["state"].items()}
                restored.optimizer.load_state_dict(optimizer)
        restored.generator.set_state(restored.torch.tensor(state["generator"], dtype=restored.torch.uint8))
        restored.updates = model["updates"]
        restored.metrics = deepcopy(model["metrics"])
        restored.episodes = deepcopy(state["episodes"])
        restored.trajectory = deepcopy(state["trajectory"])
        self.__dict__.update(restored.__dict__)

    def evaluation_copy(self):
        clone = type(self)({**self.settings, "seed": self.seed})
        if self.map is not None:
            clone._initialize(self.map)
            clone.networks.load_state_dict(self.networks.state_dict())
        return clone
