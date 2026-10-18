"""AIのモデル・optimizer・乱数・経験を保存する。環境やWebアプリには依存しない。"""

from dataclasses import asdict


class PPOCheckpointMixin:
    """PPOの実装を共有せず、学習状態のシリアライズだけを共通化する。"""

    def evaluation_copy(self):
        return type(self).from_export(self.export())

    def checkpoint(self) -> dict:
        if self._pending is not None:
            raise ValueError("行動のlearn()が終わってから状態を保存してください。")

        def encode(value):
            if self._torch is not None and isinstance(value, self._torch.Tensor):
                return {"tensor_dtype": str(value.dtype).removeprefix("torch."),
                        "tensor_values": value.detach().cpu().tolist()}
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(item) for item in value]
            return value

        return {"schema_version": 1, "model": self.export(),
                "rollout": [asdict(item) for item in self._rollout],
                "optimizer": None if self._optimizer is None else encode(self._optimizer.state_dict()),
                "generator": None if self._generator is None else self._generator.get_state().tolist()}

    def load_checkpoint(self, state: dict) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("対応していないPPO学習状態の形式です。")
        restored = type(self).from_export(state["model"])

        def decode(value):
            if isinstance(value, dict) and set(value) == {"tensor_dtype", "tensor_values"}:
                return restored._torch.tensor(value["tensor_values"],
                                               dtype=getattr(restored._torch, value["tensor_dtype"]))
            if isinstance(value, dict):
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value

        if state["optimizer"] is not None:
            optimizer = decode(state["optimizer"])
            optimizer["state"] = {int(key): value for key, value in optimizer["state"].items()}
            restored._optimizer.load_state_dict(optimizer)
            restored._generator.set_state(restored._torch.tensor(state["generator"],
                                                                 dtype=restored._torch.uint8))
        restored._rollout = [self._transition_type(**{**item, "state": tuple(item["state"])})
                             for item in state.get("rollout", [])]
        restored.settings = self.settings.copy()
        self.__dict__.update(restored.__dict__)

