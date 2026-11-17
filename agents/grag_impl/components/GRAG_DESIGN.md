# GRAG 設計書

この文書は `agents.grag.GRAG`（実装: `agents/grag_impl/grag.py`）の現行実装を説明する。別の `grag_sgcrl.SGCRL` は対象外。2026-11-17時点。

## 目的と構成

GridWorldの観測から4方向の行動を選び、成功経験を用いた時間重み付き対照学習で状態行動と目標の対応を学ぶ。得られたスコアからQ値とアドバンテージを計算し、PPO-ClipでActorを更新する。PRTSの対照損失を参考にしたオンライン学習であり、論文の学習手順全体を再現したものではない。

```mermaid
flowchart LR
    Env[GridWorld] -->|観測| Actor[Actor]
    Actor -->|行動| Env
    Env -->|遷移| Mem[memory]
    Mem -->|成功区間| CRL[対照学習]
    CRL --> SA[状態行動エンコーダー]
    CRL --> Goal[目標エンコーダー]
    SA --> Q[Q = exp(内積)]
    Goal --> Q
    Mem --> Q
    Q --> Adv[アドバンテージ]
    Adv --> PPO[PPO Actor更新]
    PPO --> Actor
```

| ファイル | 役割 |
| --- | --- |
| `../grag.py` | エージェント、経験収集、損失、Q、Actor更新 |
| `networks.py` | Actor、Critic、2つのエンコーダーの構築 |
| `language.py` | Ruriによる固定言語特徴の抽出 |
| `../../grag.py` | 公開入口 |

## 観測と行動

観測辞書の `rows`, `cols`, `obstacles`, `goal`, `position` を `[3, rows, cols]` のfloat32テンソルにする。チャネル順は障害物、ゴール、現在位置。行動は `up`, `down`, `left`, `right` の4種類である。

`choose_action` は初回にネットワークを生成し、Actorのロジットから行動をサンプルする。`greedy=True` なら最大確率の行動を返す。`action_probabilities` は初期化後の各行動確率を返す。目標文はマップとゴールから `generate_goal_prompt` で生成する。

## ネットワーク

- ActorとCriticは全観測を平坦化し、幅 `ppo_hidden_size` のTanh隠れ層2層を通す。Actorは4行動のロジット、Criticはスカラーを出力する。
- 状態行動エンコーダー `phi` は障害物・位置チャネルと行動one-hotを受け取る。幅512の4層、LayerNorm、SiLU、64次元出力。
- 目標エンコーダー `psi` は384次元の言語特徴を受け取り、同じ構成で64次元を出力する。
- `language.py` は `cl-nagoya/ruri-v3-70m` を初回使用時にロードする。RuriのL2正規化済み特徴は固定し、目標文ごとにキャッシュする。`psi` の出力はキャッシュしない。

Criticは遷移記録時の値の取得には使うが、現在の `_update_grag` はCriticを学習しない。Actorのadvantageには対照学習によるQを使う。

## 経験と更新条件

`learn` は各遷移の状態、行動、旧方策の対数確率、価値、報酬、次状態価値、ゴール到達、目標文を `memory` に保持する。`end_episode` は最終遷移に試行終了を記録する。

ゴール到達時、保持中の `reached_goal=True` の数が `grag_positive_update_count` 以上なら更新を試みる。成功した軌道は1本につき1件と数える。失敗・未完了の遷移もActor用経験として保持する。経験に含まれるすべての目標文に成功参照がなければ更新せず保持する。Actorの更新に成功した場合、memory全体を消去する。成功リプレイはない。

## 学習処理

1. `_crl_transition_metadata` が各遷移の目標文と成功までの残りステップを求める。ゴール到達で終わる同一目標の区間が正例。失敗・未完了区間は対象外。
2. `_compute_crl_loss` は正例について `f(s,a,l)=phi(s,a)ᵀpsi(l)` を計算し、状態行動→言語と、言語→状態行動の交差エントロピーを合計する。正例の時間重みは `gamma^(T-t)`。成功遷移は残り0で重み1。対象正例が2件未満、または係数0なら損失を返さない。
3. 損失があれば両エンコーダーをAdamで1回更新する。勾配ノルム上限は0.5。
4. 全経験について全4行動のスコアを計算し、補正定数を使わず `Q=exp(f)` とする。
5. 現在のActor確率で `V(s)=Σa π(a|s)Q(s,a)`、選択行動の `A=Q(s,a)-V(s)` を求める。全経験のadvantageを標準化する。
6. 旧対数確率と固定advantageを使い、シャッフルしたミニバッチでPPO-Clipとエントロピー項によりActorをAdam更新する。勾配ノルム上限は0.5。全epoch合計のoptimizer stepは `grag_actor_max_steps` 以下。

単一目標では状態行動→言語の分類は1クラスとなり、その方向の損失は0。逆方向の時間重み付き損失は残る。成功区間に含まれない状態・行動のQはエンコーダーの汎化に依存する。

## 主な設定

以下は `GRAG(settings)` に渡す設定の既定値。UIや呼び出し元が値を指定すると上書きされる。

| キー | 既定値 | 役割 |
| --- | --- | --- |
| `grag_positive_update_count` | 16 | 更新を試みる成功軌道数。1以上の整数 |
| `grag_actor_max_steps` | 16 | 1回の更新でのActor optimizer step上限。1以上の整数 |
| `ppo_learning_rate` | 0.00025 | Actor学習率。CRL学習率の省略時にも使用 |
| `ppo_hidden_size` | 64 | Actor・Criticの隠れ層幅 |
| `ppo_epochs` | 4 | Actorの最大epoch数。`update_epochs` も指定可能 |
| `ppo_minibatch_size` | 64 | Actorミニバッチ数。`minibatch_size` も指定可能 |
| `ppo_clip_range` | 0.2 | PPOのclip幅。`clip_coef` も指定可能 |
| `ppo_entropy_coef` | 0.01 | Actorのエントロピー係数。`ent_coef` も指定可能 |
| `crl_learning_rate` | `ppo_learning_rate` | 対照エンコーダー学習率 |
| `crl_coef` | 1.0 | 対照損失係数 |
| `crl_gamma` | `ppo_gamma` 等、既定0.99 | 成功までの時間重みの割引率 |
| `seed` | 未指定 | 初期化・行動選択・ミニバッチ順の乱数 |

`crl_actor_learning_rate`, `crl_hidden_size`, `crl_min_replay_size` は現行GRAGでは使用しない。`ppo_rollout_steps` も更新条件には使わない。

## 制約と検証

`export()` は現在空の辞書を返し、GRAG本体に学習状態の保存・復元は実装されていない。成功例の収集に時間がかかる場合、経験が増え続ける。

リポジトリ直下から次を実行する。

```powershell
python -m unittest discover -s agents/grag_impl/tests -t . -v
```

更新条件とPRTS損失に絞る場合は `python -m unittest agents.grag_impl.tests.test_positive_update agents.grag_impl.tests.test_prts_loss`。テストの通過はオンライン学習の安定性を保証するものではない。
