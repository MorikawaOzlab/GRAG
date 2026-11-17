"""Loss functions used by GRAG."""

import torch
from torch import nn


def compute_crl_loss(self, transitions, state_action_encoder, encode_language):
    """PRTS式(10)〜(12)に基づく双方向対照損失を計算する。

    成功エピソードのみを使用する。
    時間重みは gamma ** (T - t) とする。
    同じgoal_promptは同じタスクとして扱う。

    Returns:
        crl_coef * (L_sa_to_l + L_l_to_sa)
        学習可能な経験が不足する場合はNone。
    """
    if not transitions:
        return None

    coef = float(self.settings.get("crl_coef", 1.0))
    gamma = float(self.settings.get("crl_gamma", self.gamma))

    if coef == 0.0:
        return None
    if not 0.0 < gamma <= 1.0:
        raise ValueError("crl_gammaは0より大きく1以下にしてください。")

    if self.sa_encoder is None or self.goal_encoder is None:
        return None

    # 1. 成功エピソードから経験と時間重みを抽出
    samples = []
    remaining_steps = []

    episode = []

    def append_successful_episode(episode):
        if not episode or not episode[-1].reached_goal:
            return

        length = len(episode)

        for i, transition in enumerate(episode):
            if transition.goal_prompt is None:
                continue

            samples.append(transition)
            remaining_steps.append(length - i - 1)

    for transition in transitions:
        episode.append(transition)

        if transition.reached_goal or transition.episode_end:
            append_successful_episode(episode)
            episode = []

    # 未完了の最後のエピソードは使用しない
    if len(samples) < 2:
        return None

    # 2. 重複しないゴール文を取得
    goals = list(dict.fromkeys(t.goal_prompt for t in samples))

    # 3. 状態・行動の特徴ベクトルを生成
    states = torch.stack([t.state for t in samples])
    actions = torch.tensor([t.action for t in samples])

    sa_features = state_action_encoder(
        states,
        actions,
        num_actions=self.CRL_NUM_ACTIONS,
    )

    phi = self._encode_crl_features(
        self.sa_encoder, sa_features
    )

    # 4. 言語特徴を生成
    # encode_language(prompt) が1次元Tensorを返す想定
    language_features = []

    for goal in goals:
        if goal not in self._goal_embedding_cache:
            embedding = encode_language(goal)
            self._goal_embedding_cache[goal] = (
                embedding.detach().flatten().cpu()
            )

        language_features.append(
            self._goal_embedding_cache[goal]
        )

    language_features = torch.stack(language_features)

    psi = self._encode_crl_features(
        self.goal_encoder, language_features
    )

    # 5. 内積スコアを計算
    # scores: [経験数, ゴール数]
    scores = phi @ psi.T

    device = scores.device
    dtype = scores.dtype

    goal_to_index = {
        goal: i for i, goal in enumerate(goals)
    }

    target = torch.tensor(
        [goal_to_index[t.goal_prompt] for t in samples],
        device=device,
        dtype=torch.long,
    )

    # 6. 時間重み gamma ** (T - t)
    remaining = torch.tensor(
        remaining_steps, device=device, dtype=dtype
    )
    weights = torch.pow(
        torch.tensor(gamma, device=device, dtype=dtype),
        remaining,
    )

    # 7. (s,a) -> l の対照損失: PRTS式(11)
    # 同じゴール文は1列にまとめる
    log_probs_sa = torch.log_softmax(scores, dim=1)

    positive_log_probs = log_probs_sa.gather(
        1, target.unsqueeze(1)
    ).squeeze(1)

    loss_sa_to_l = -(
        weights * positive_log_probs
    ).mean()

    # 8. l -> (s,a) の対照損失: PRTS式(12)
    # 同じゴールの全経験が正例
    log_probs_l = torch.log_softmax(scores.T, dim=1)

    loss_l_to_sa = scores.new_zeros(())

    for goal_index in range(len(goals)):
        mask = target == goal_index

        # 同一ゴール内で時間重みを正規化
        positive_weights = weights[mask]
        q = positive_weights / positive_weights.sum()

        goal_loss = -(
            q * log_probs_l[goal_index, mask]
        ).sum()

        # PRTS式(12)の経験単位の平均に対応
        loss_l_to_sa = loss_l_to_sa + (
            mask.sum() / len(samples)
        ) * goal_loss

    # 9. 双方向損失を合計
    return coef * (loss_sa_to_l + loss_l_to_sa)



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
