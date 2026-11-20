
import torch
import torch.nn.functional as F
import torch.nn as nn


def compute_crl_loss(
    self,
    sa_embeddings: torch.Tensor,
    goal_embeddings: torch.Tensor,
) -> torch.Tensor:
    """CPPOのInfoNCE対照損失を計算する。

    Args:
        sa_embeddings: 状態・行動表現 [N, D]
        goal_embeddings: 正例ゴール表現 [N, D]

    同じインデックスを正例とし、
    異なるインデックスを負例とする。

    Returns:
        スカラーの対照学習損失。
    """
    # 全組合せの内積
    logits = sa_embeddings @ goal_embeddings.T

    # 対角成分を正例とする
    labels = torch.arange(
        logits.shape[0],
        device=logits.device,
    )

    # InfoNCE損失
    loss = F.cross_entropy(logits, labels)

    return loss

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
