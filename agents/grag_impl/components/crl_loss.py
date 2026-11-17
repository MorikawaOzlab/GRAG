"""GRAG: 成功・失敗エピソードを用いる時間重み付き対照損失。"""

import torch
from torch import nn
from torch.nn import functional as F


def compute_crl_loss(self, transitions, state_action_encoder, encode_language):
    """成功軌道を正例、失敗軌道を負例として学習する実験用損失。

    各エピソードの終端までの残り時間に gamma ** remaining の重みを与える。
    同じ goal_prompt でも学習可能。失敗軌道内の有益な行動も負例と
    みなすため、これはQ値の不偏推定を保証しない。

    設定値:
        crl_coef: 全体の係数 (既定1.0)
        crl_gamma: 時間割引率 (既定self.gamma)
        crl_negative_coef: 失敗軌道の損失係数 (既定1.0)
    """
    if not transitions:
        return None

    coef = float(self.settings.get("crl_coef", 1.0))
    gamma = float(self.settings.get("crl_gamma", self.gamma))
    negative_coef = float(self.settings.get("crl_negative_coef", 1.0))

    if coef == 0.0:
        return None
    if not 0.0 < gamma <= 1.0:
        raise ValueError("crl_gammaは0より大きく1以下にしてください。")
    if negative_coef < 0.0:
        raise ValueError("crl_negative_coefは0以上にしてください。")
    if self.sa_encoder is None or self.goal_encoder is None:
        return None

    samples = []
    labels = []
    remaining_steps = []
    episode = []

    def append_episode(steps):
        if not steps:
            return
        successful = bool(steps[-1].reached_goal)
        length = len(steps)
        for i, transition in enumerate(steps):
            if transition.goal_prompt is None:
                continue
            samples.append(transition)
            labels.append(float(successful))
            remaining_steps.append(length - i - 1)

    for transition in transitions:
        episode.append(transition)
        if transition.reached_goal or transition.episode_end:
            append_episode(episode)
            episode = []
    # 未完了の最後のエピソードは含めない。

    if not samples:
        return None
    if not any(labels):
        # 正例がない場合は誤って一方的にスコアを下げることを避ける。
        return None

    states = torch.stack([t.state for t in samples])
    actions = torch.tensor([t.action for t in samples], dtype=torch.long)
    sa_features = state_action_encoder(
        states, actions, num_actions=self.CRL_NUM_ACTIONS,
    )
    phi = self._encode_crl_features(self.sa_encoder, sa_features)

    goals = list(dict.fromkeys(t.goal_prompt for t in samples))
    language_features = []
    for goal in goals:
        if goal not in self._goal_embedding_cache:
            embedding = encode_language(goal)
            self._goal_embedding_cache[goal] = embedding.detach().flatten().cpu()
        language_features.append(self._goal_embedding_cache[goal])
    language_features = torch.stack(language_features)
    psi = self._encode_crl_features(self.goal_encoder, language_features)

    goal_to_index = {goal: i for i, goal in enumerate(goals)}
    target = torch.tensor(
        [goal_to_index[t.goal_prompt] for t in samples],
        device=phi.device, dtype=torch.long,
    )
    # 経験と、その経験が属するゴールのスコアのみ比較する。
    scores = (phi * psi[target]).sum(dim=-1)
    dtype, device = scores.dtype, scores.device
    labels = torch.tensor(labels, dtype=dtype, device=device)
    remaining = torch.tensor(remaining_steps, dtype=dtype, device=device)
    weights = torch.pow(torch.as_tensor(gamma, dtype=dtype, device=device), remaining)

    # 正例はスコア上昇、負例はスコア低下方向へ更新する。
    # softplus(-score)=-log(sigmoid(score))、softplus(score)=-log(1-sigmoid(score))
    pos_mask = labels == 1
    neg_mask = ~pos_mask
    pos_loss = (weights[pos_mask] * F.softplus(-scores[pos_mask])).sum() / weights[pos_mask].sum().clamp_min(1e-8)
    if neg_mask.any() and negative_coef > 0.0:
        neg_loss = (weights[neg_mask] * F.softplus(scores[neg_mask])).sum() / weights[neg_mask].sum().clamp_min(1e-8)
    else:
        neg_loss = scores.new_zeros(())

    return coef * (pos_loss + negative_coef * neg_loss)


def compute_ppo_loss(
    actor: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
    ent_coef: float,
) -> torch.Tensor:
    """Criticを使用しないPPOの方策損失を計算する。"""

    # Actorの出力
    logits = actor(states)
    dist = torch.distributions.Categorical(logits=logits)

    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    # 新旧方策の確率比
    ratio = torch.exp(new_log_probs - old_log_probs)

    # PPOのクリップ付き方策損失
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(
        ratio,
        1.0 - clip_coef,
        1.0 + clip_coef,
    )

    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

    # エントロピー正則化
    loss = policy_loss - ent_coef * entropy

    return loss
