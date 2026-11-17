# GRAG: Single-goal Contrastive RL

<<<<<<< Updated upstream
`agents/grag.py` を公開入口、`grag.py` を本体として、Liu et al. の
[A Single Goal Is All You Need (ICLR 2025)](https://arxiv.org/abs/2408.05804)
の学習則を既存のグリッド迷路に適用します。
これは**離散迷路での手法再現**です。論文と同じ連続制御環境・実験結果の再現ではありません。
=======
Actor の更新は、1回の `_update_grag()` につき全 epoch 合計で最大16ミニバッチです。
`grag_actor_max_steps`（1以上の整数、既定値16）で上限を変更できます。
経験数と `ppo_epochs` から決まる更新回数が上限未満なら、その回数で終了します。
各 epoch で経験をシャッフルし、上限に達した時点で Actor の更新を終了します。

GridWorld用のGoal-Reachability-Aware Gatingエージェント。
公開入口は `from agents.grag import GRAG`。
>>>>>>> Stashed changes

## 論文との対応

<<<<<<< Updated upstream
| 項目 | 現在の実装 |
| --- | --- |
| Algorithm 1 の経験収集 | `observation["goal"]`（迷路の最終ゴール）を常にActorへ渡す |
| 状態 `s` | 現在座標 `(行, 列)`。迷路の構造はエージェントごとに固定 |
| 目標 `g` | 座標 `(行, 列)`。左上は `(0, 0)` |
| Critic | 非正規化の `phi(s,a)` と `psi(g)` の内積。各表現は64次元 |
| 式(3) | 行方向のInfoNCE + `0.01 * logsumexp(scores)^2` |
| 正例 | 同じ試行の将来状態を、割引係数に基づく幾何分布で抽出 |
| 負例 | バッチ内の将来状態を周辺分布のサンプルとして利用 |
| 式(4) | 将来目標・シャッフルした目標でActorを学習。内積スコアとエントロピーを最大化 |
| エントロピー係数 | `alpha * (entropy - target_entropy)` で自動調整 |
| 推論 | 最終ゴールを条件にActorから行動を選ぶ。評価時は最大確率の行動 |
=======
- `grag.py`: 観測変換、行動選択、経験管理、対照学習・Q・アドバンテージ計算、PPO Actor更新。
- `grag_sgcrl.py`: 環境非依存の状態ゴール版SGCRLとGridWorld接続アダプター。
- `components/language.py`: Ruri v3 70mの遅延読み込みと固定言語特徴の抽出。
- `components/networks.py`: Actor、Critic、状態行動・目標エンコーダーの構築。
- `grag_interface.py`: エージェントのインターフェースひな形。
- `tests/test_prts_loss.py`: 現行の対照損失、Q、アドバンテージ、Actor更新のテスト。
>>>>>>> Stashed changes

Actorの学習目標は最終ゴールだけではありません。経験収集の目標を一つに固定することと、
経験中の将来状態を学習目標として使うことを区別します。最終ゴールが未訪問でも学習します。
報酬引数は共通APIとの互換のため受け取りますが、損失には一切使いません。
PPO、GAE、報酬を予測するCritic、最終ゴールの訪問確認による更新制限はありません。
衝突で静止した遷移や同じ座標への再訪も将来状態分布の一部として残します。

<<<<<<< Updated upstream
参照した公式実装:
[learning.py](https://github.com/graliuce/sgcrl/blob/main/contrastive/learning.py)、
[builder.py](https://github.com/graliuce/sgcrl/blob/main/contrastive/builder.py)、
[networks.py](https://github.com/graliuce/sgcrl/blob/main/contrastive/networks.py)。
=======
現行実装は成功区間の経験を対照学習の正例として使い、Q = exp(状態行動表現と目標表現の内積)から計算した
アドバンテージでPPOのActorを更新する。対象目標の成功参照が不足している間は
経験を保持して更新を待つ。`export()`は現在、空の辞書を返す。
>>>>>>> Stashed changes

## 離散迷路への適用と制限

- 論文は連続行動です。本実装は4方向のカテゴリカル方策とし、式(4)の行動期待値を4行動の総和で正確に計算します。
- 同一エージェントで迷路構造を変更することはできません。固定迷路内の現在座標と目標座標を入力します。障害物の位置や最短経路は学習入力に与えません。
- 共通GridWorldの仕様により、ゴール到達または行動回数上限で試行を終了します。元論文の連続制御タスク・固定時間のデータ分布とは異なります。
- 有限試行内の将来状態を割引付きで抽出します。元の実装と同様に、試行をまたいで将来状態を抽出しません。
- `crl_action_scores()` は内積スコアを返します。絶対的な到達確率ではありません。推論にはActorを使います。
- 学習の成功率とgreedy評価の成功は別の指標です。小規模な動作確認だけで長い迷路の探索性能や論文の結果を再現したとは判断できません。

## 設定

`data/training_settings.json` の `crl_*` 設定、または `GRAG(settings)` で指定します。
画面の `ppo_*` 設定やQ学習の割引率は、現在のGRAGには適用しません。

| 設定 | 既定値 |
| --- | ---: |
| `crl_device` | `"auto"`（CUDAが利用可能ならGPU） |
| `crl_gamma` | 0.99 |
| `crl_learning_rate` | 0.0003 |
| `crl_actor_learning_rate` | 0.0003 |
| `crl_batch_size` | 256 |
| `crl_hidden_size` | 256（ReLUの隠れ層2層） |
| `crl_repr_dim` | 64 |
| `crl_min_replay_size` | 10000 |
| `crl_replay_capacity` | 1000000 |
| `crl_updates_per_step` | 1.0 |
| `crl_target_entropy` | 0.0 |

最初の10000遷移は一様ランダムで収集します。完了した試行の経験がこの数に達してから更新し、
以降は試行の終了ごとに、その試行の遷移数に応じた回数の勾配更新を行います。
バッチ256の既定値では、1遷移追加あたり1更新は論文の samples-per-insert 256 に相当します。
更新中は次の環境遷移を待つため、長い試行では終了時の処理が重くなります。

Actor・両対照エンコーダー・損失計算・勾配更新・alphaの更新は選択したデバイス上で実行します。
経験バッファーと乱数生成はCPUに置き、学習バッチをGPUへ転送します。
`crl_device="cuda"` / `"cuda:0"` でGPUを明示指定、`"cpu"` でCPUを指定できます。
GPUを明示指定して利用できない場合はエラーになります。`"auto"` はGPUがなければCPUを使います。
小さなネットワークではGPUへの転送や起動コストが大きく、CPUより高速になるとは限りません。

## 保存・検証

`export()/from_export()` はActorと対照表現を保存・復元します。
`checkpoint()/load_checkpoint()` は経験、途中の試行、乱数、Optimizer、alphaも保存・復元します。
保存するテンソルはCPU形式なので、CPUとGPUをまたいで復元できます。
`load_checkpoint()` は復元先のエージェントのデバイスを使います。
推論用モデルは `GRAG.from_export(model, device="cpu")` のように復元先を指定できます。
旧PPO系GRAGのモデル・チェックポイントとは互換性がありません。新しい学習を開始してください。

```powershell
python -m pip install -r agents/grag_impl/requirements.txt
python -m unittest -v agents.grag_impl.tests.test_grag_update agents.grag_impl.tests.test_training_settings
python -m unittest -v agents.grag_impl.tests.test_gpu
python -m scripts.verify_sgcrl --episodes 1000
python -m scripts.verify_sgcrl --episodes 100 --seeds 7 --device cuda
```

<<<<<<< Updated upstream
検証スクリプトは短時間の学習確認をするため、ネットワーク・バッチ・初期収集を小さくしています。
既定ではGPUがあればGPUを使います。CPUを使う場合は `--device cpu` を指定します。
論文のベンチマーク設定とは異なります。3シードの成功率とgreedy評価を別々に出力します。
具体的な不一致・修正内容・確認結果は [検証記録](SGCRL_VERIFICATION.md) を参照してください。

以前のPPO混合実装は `grag_ppo_reference.py`、その説明は `PPO_REFERENCE.md` に保存しています。
旧仕様を検査する `test_goals.py`、`test_ppo.py`、`test_cleanrl_grid.py` は現在のSGCRLを対象としません。
独立したPPOエージェントの実装は `agents/ppo_agent_impl/` です。
=======
旧GRAG・SGCRLの参考実装と旧API用テストは削除済み。
独立したQ学習・PPOはGridWorldから使われるため、`agents/`内に維持する。

## 将来の(s,a)-g-l版への拡張

`grag_sgcrl.SGCRL`を継承し、経験管理・将来ゴール抽出・状態ゴール学習を
再利用する。`grag_sgcrl.GRAG`は同じクラスの互換名。
`grag_sgcrl.SGCRL.initialize_networks(state_dim, goal_dim, action_dim, discrete=False)`
で、環境に依存せずネットワークを初期化できる。
論文Appendix Bと著者実装に沿って、既定ではActorと2つの対照エンコーダーを
隠れ層(256, 256)・ReLU、表現次元64で構築する。
状態とゴールは独立した特徴次元を指定できる。Actorは結合した`[s, g]`を
受け取り、連続行動ではtanh変換した対角Gaussian分布、`discrete=True`では
指定した行動数のCategorical分布を返す。連続行動は`[-1, 1]`に正規化されており、
環境の行動範囲への変換はアダプター側で扱う。
状態行動エンコーダーは`[s, a]`、ゴールエンコーダーは`g`を受け取る。
観測の前処理・特徴抽出、離散行動のone-hot化は環境側で行う。
同じ仕様で再度呼んでも重みを保持し、異なる次元・行動種別にはエラーを返す。
seedによる再現性を保ち、外部のPyTorch乱数状態は変更しない。
`choose_action(state, goal, greedy=True)`は離散方策では最大確率の行動、
連続方策ではGaussianの平均をtanh変換した行動を返す。`greedy=False`（既定値）
では初期収集中は一様ランダム、その後はActorの分布からサンプルする。
バッチ入力にも対応し、greedy時は乱数を使わない。
分布の`sample()`や連続方策の`rsample()`も利用できる。
`learn()`は遷移を保持し、終端・時間制限または`end_episode()`で軌跡を確定する。
状態とゴールの特徴次元が異なる場合は、次状態をゴール表現に射影した
`next_goal`も渡す。成功・失敗どちらの軌跡も使い、報酬は受け取らない。
同じ軌跡内の将来状態を割引幾何分布で抽出し、行方向InfoNCEと
`0.01 * logsumexp(logits)**2`でエンコーダーを更新する。
Actorは正例ゴールとバッチをずらしたゴールを半数ずつ使い、
内積スコアとエントロピーを最大化する。エントロピー係数もAdamで学習する。

```python
import torch
from agents.grag_impl.grag_sgcrl import SGCRL

agent = SGCRL({"seed": 7})
agent.initialize_networks(state_dim=11, goal_dim=3, action_dim=2)
state, goal = torch.zeros(11), torch.zeros(3)
distribution = agent.actor(torch.cat((state, goal), dim=-1))
action = distribution.sample()  # shape: (2,)
```

- `encode_state_actions()`がphi(s,a)、`encode_goals()`が共有するpsi(g)。
- `state_goal_scores()`は未正規化の内積を返す。到達確率への補正は含めない。
- 言語対応版でchi(l)を追加し、`_additional_encoder_losses()`からg-l等の損失を返す。
- `_encoder_parameters()`を拡張し、学習するchiのパラメーターも登録する。
  固定の言語モデルを使う場合も、その後の学習可能な射影層は登録する。
- `critic_loss()`では(s,a)-gと追加損失を合計し、`update()`で一度のbackward/stepを行う。
- ゴールと言語の対応データは言語対応版で管理する。同じゴールに複数の言語表現が
  対応する場合など、正例の定義は追加損失側で扱う。言語のない経験もSGCRLで学習する。
- 言語から行動する仕組みは別途設計する。g-lの表現を揃えるだけで、
  現在の状態ゴール入力のActorが言語を直接受け取れるわけではない。

将来の言語対応版はSGCRLを利用する向きで依存させ、SGCRLから言語対応版を
importしない。現行`grag.py`は当面、環境入出力と補助処理の再利用元として維持する。

GRAG updates when at least `grag_positive_update_count` successful trajectories have accumulated (default: 128; positive integer). Each goal-reaching trajectory counts once; failed episodes do not. If a goal lacks a successful reference, experience is retained until one arrives. `ppo_rollout_steps` does not control GRAG updates.
# GRAGの更新条件

GRAGは成功軌道を `grag_positive_update_count` 本収集してから、対照エンコーダーとActorを順に更新します。初期値は8本です。Web UIの「GRAGの1更新に必要な成功軌道数」で変更でき、変更は「新しい学習」から適用されます。成功していない軌道だけでは更新しません。失敗軌道は古いものから破棄し、直近の失敗軌道を少なくとも1本保持します。

>>>>>>> Stashed changes
