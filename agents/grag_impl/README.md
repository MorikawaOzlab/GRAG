# GRAG: Single-goal Contrastive RL

`agents/grag.py` を公開入口、`grag.py` を本体として、Liu et al. の
[A Single Goal Is All You Need (ICLR 2025)](https://arxiv.org/abs/2408.05804)
の学習則を既存のグリッド迷路に適用します。
これは**離散迷路での手法再現**です。論文と同じ連続制御環境・実験結果の再現ではありません。

## 論文との対応

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

Actorの学習目標は最終ゴールだけではありません。経験収集の目標を一つに固定することと、
経験中の将来状態を学習目標として使うことを区別します。最終ゴールが未訪問でも学習します。
報酬引数は共通APIとの互換のため受け取りますが、損失には一切使いません。
PPO、GAE、報酬を予測するCritic、最終ゴールの訪問確認による更新制限はありません。
衝突で静止した遷移や同じ座標への再訪も将来状態分布の一部として残します。

参照した公式実装:
[learning.py](https://github.com/graliuce/sgcrl/blob/main/contrastive/learning.py)、
[builder.py](https://github.com/graliuce/sgcrl/blob/main/contrastive/builder.py)、
[networks.py](https://github.com/graliuce/sgcrl/blob/main/contrastive/networks.py)。

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

検証スクリプトは短時間の学習確認をするため、ネットワーク・バッチ・初期収集を小さくしています。
既定ではGPUがあればGPUを使います。CPUを使う場合は `--device cpu` を指定します。
論文のベンチマーク設定とは異なります。3シードの成功率とgreedy評価を別々に出力します。
具体的な不一致・修正内容・確認結果は [検証記録](SGCRL_VERIFICATION.md) を参照してください。

以前のPPO混合実装は `grag_ppo_reference.py`、その説明は `PPO_REFERENCE.md` に保存しています。
旧仕様を検査する `test_goals.py`、`test_ppo.py`、`test_cleanrl_grid.py` は現在のSGCRLを対象としません。
独立したPPOエージェントの実装は `agents/ppo_agent_impl/` です。
