# SGCRL検証記録（2026-11-09）

対象: 現在のグリッド迷路での「A Single Goal Is All You Need」の学習則の適用。
元論文のPoint Spiral・Sawyer環境の実験結果を再現したという記録ではありません。

## 以前の実装で確認した不一致

1. 報酬からGAEを計算し、PPOでActorを更新していた。論文の報酬なしの式(4)と異なる。
2. 最終ゴールの正例がなければ対照Qを使用せず、GAEへ戻っていた。
   最終ゴール未訪問の短い試行で、対照更新が1回進んでもadvantageの選択は `gae` だった。
3. Actorを最終ゴールだけで学習していた。論文では経験中の将来状態もActorの学習目標にする。
4. 到達した座標を重複除去し、最短到達時間を採用していた。
   論文の割引付き将来状態サンプリングと周辺分布の頻度を維持できていなかった。
5. 対称な複数正例損失を用い、式(3)のLogSumExp正則化がなかった。
6. 参照バッファーから補正した指数Q値を使用していた。
   論文のActorは非正規化表現の内積スコアを直接最大化する。

以前の実装は `grag_ppo_reference.py` に保存し、現行の学習経路から外した。

## 現行実装の確認

`test_grag_update.py` / `test_training_settings.py` は以下を検査する。

- 式(3)が独立したスカラー計算と一致し、数値微分でも勾配が一致する。
- 式(4)の離散期待値が一致し、高スコアの行動確率を上げる向きの勾配が出る。
- 同じ経験で報酬を0から999へ変えても、全モデル・Optimizer・乱数状態が一致する。
- 最終ゴール未訪問でもActorと両エンコーダーが更新される。
- Actor更新に将来目標とシャッフルした目標を入力し、経験収集には迷路の最終ゴールを入力する。
- 幾何分布による将来状態の抽出頻度が一致し、エピソード境界を越えない。
- 衝突と再訪をバッファーに残す。
- greedy推論がActorの最大確率行動を返し、学習用乱数を消費しない。
- JSON保存・途中再開後の行動、更新後重み、Optimizer、alpha、乱数が連続実行と完全一致する。
- 学習済みexportからの確率的推論がActorを使い、初期ランダム収集へ戻らない。
- 初期化前・seed未指定のチェックポイントからも同じ初期重みを復元する。

共通環境・API・PPO独立実装・ライブ学習・行動確率表示を含め、次の69テストが成功。

```powershell
python -m unittest agents.grag_impl.tests.test_grag_update agents.grag_impl.tests.test_training_settings gridworld.tests.test_app gridworld.training_runtime.tests.test_session gridworld.training_runtime.tests.test_api gridworld.training_runtime.tests.test_ppo_session gridworld.training_runtime.tests.test_live_training gridworld.training_runtime.tests.test_policy_inspection
```

画面のJavaScriptは `deno check --no-config --no-lock` で構文確認した。
GRAG選択時の設定生成も実行し、`crl_gamma`・学習率・隠れ層幅・初期収集数が渡り、
PPO設定を送らないことを確認した。画面操作による目視確認は行っていない。

## 小規模学習確認

実行: `python -m scripts.verify_sgcrl --episodes 1000`

2行3列、開始 `(0,0)`、ゴール `(1,2)`、壁 `(0,1)`、最大16行動。
隠れ層幅64、表現次元16、バッチ32、初期ランダム収集64遷移、他は既定値。
この小規模設定は論文の設定とは異なる。

| Seed | 学習の遷移数 | 勾配更新数 | 最後の100試行の到達率 | greedy到達 | greedyの行動数 |
| ---: | ---: | ---: | ---: | --- | ---: |
| 7 | 5062 | 5000 | 100% | 成功 | 3 |
| 42 | 4976 | 4926 | 100% | 成功 | 3 |
| 123 | 5093 | 5030 | 100% | 成功 | 3 |

短い300試行での予備確認ではgreedy失敗もあった。学習中の確率的な成功率だけで、
決定的方策も完成したと判断しない。大きな迷路での探索性能、最初の成功までの遷移数、
元論文と同じ環境・5シード・学習予算の比較は未検証。

## GPU対応の追加確認

PyTorch `2.14.1+cu130`、CUDA 13.0、NVIDIA GeForce RTX 5070 Tiで確認。
`crl_device` の既定値は `auto` とし、この環境では `cuda:0` を選択する。

- Actorと両エンコーダーの入力・重み・勾配、alpha、AdamのモーメントがCUDA上にあることを検査。
- CUDAチェックポイントから同じGPUで再開した更新結果が、連続実行と完全一致することを検査。
- CPUチェックポイントからGPUで学習を続行し、その状態をCPUでも再開できることを検査。
- GPU非搭載を模した環境でも、GPUで保存したモデルをCPUへ明示指定して復元・更新できることを検査。
- GPUで100試行・588更新の実学習を実行し、損失が有限であることを確認。

GPU検査は `python -m unittest agents.grag_impl.tests.test_gpu` で実行できる。
GPU検査7件を含む現行の回帰テスト76件が成功した。
小さなネットワークのため、この確認からGPUによる速度向上は主張しない。

## 一次資料（参照リンク）

- [論文、式(3)(4)・Algorithm 1・Appendix B](https://proceedings.iclr.cc/paper_files/paper/2025/file/c34ddd05eb089991f06f3c5dc36836e0-Paper-Conference.pdf)
- [公式学習実装](https://github.com/graliuce/sgcrl/blob/main/contrastive/learning.py)
- [公式データ生成・幾何分布サンプラー](https://github.com/graliuce/sgcrl/blob/main/contrastive/builder.py)
- [公式ネットワーク](https://github.com/graliuce/sgcrl/blob/main/contrastive/networks.py)
