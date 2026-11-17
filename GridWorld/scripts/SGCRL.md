# GridWorldでのSGCRL実験

SGCRL本体は`agents/grag_impl/grag_sgcrl.py`の`SGCRL`。
同じファイルの`GridWorldSGCRL`が既存の`TrainingSession`に接続する。
既存の`agents.grag.GRAG`とは独立している。
学習画面ではAIに「SGCRL」を選ぶ。初期収集の既定値は10,000遷移なので、
それまでは一様ランダムに行動する。

## 実行

リポジトリ直下で、小さな迷路を使った動作・学習チェック:

```powershell
python GridWorld/scripts/verify_sgcrl.py --episodes 300 --seeds 7 42 123 --device cpu --warmup 64 --batch-size 32 --hidden-size 64 --output output/sgcrl/quick_validation.json
```

任意の保存済みGridWorldマップで実験する例:

```powershell
python GridWorld/scripts/verify_sgcrl.py --map path/to/map.json --episodes 5000 --max-steps 200 --seeds 7 42 123 2024 2025 --output output/sgcrl/experiment.json
```

後者は既定の隠れ層256×2、表現64次元、バッチ256、初期収集10,000遷移を使う。
学習率3e-4、gamma=0.99、初期alpha=1、目標エントロピー0、リプレイ容量100万遷移。
完了エピソードの追加後、収集した遷移1件につき1回更新する。
`--updates-per-step`で更新比率を変更できる。
出力にはseedごとの学習曲線、更新数、到達率、初期・学習後のgreedy評価を保存する。
GridWorldの報酬1/0は集計にだけ使い、損失には渡さない。

## 論文との対応と相違

参照: [論文2024年公開版、式(3)(4)とAlgorithm 1、Appendix B](https://arxiv.org/html/2408.05804v1)、
[著者のネットワーク](https://github.com/graliuce/sgcrl/blob/main/contrastive/networks.py)、
[学習器](https://github.com/graliuce/sgcrl/blob/main/contrastive/learning.py)、
[将来ゴールの抽出](https://github.com/graliuce/sgcrl/blob/main/contrastive/builder.py)。
指定されたローカルPDFは本文抽出ライブラリがないため本文を抽出できず、
実装仕様は公開本文と著者コードで照合した。

- 収集と評価は同じ単一目標。学習ゴールは成功前も含む経験軌跡の将来状態から抽出する。
- 対照Criticは非正規化の内積。InfoNCEとLogSumExp正則化を使い、成功報酬や距離報酬は使わない。
- 正例は厳密に未来の状態。有限軌跡内で正規化した幾何分布を使い、終了・時間制限を越えない。
- Actor学習は正例ゴールとシャッフルしたゴールの半数ずつ。適応エントロピー係数を使う。
- 原論文の実験は連続制御。GridWorldは離散4行動なのでCategorical方策を使い、
  Actorの行動期待値を全行動で厳密に計算する。障害物方向も行動候補に残す。
- GridWorld側の特徴は障害物と位置のone-hot。状態と達成ゴールは同じ表現とし、
  これらの変換はアダプターに閉じ込める。本体は任意の特徴次元と連続・離散行動に対応する。
- PyTorchの単一学習器・エピソード確定後の更新を使う。原実装の分散JAX/Reverb、
  データセットのtranspose shuffle、ロボット環境のベンチマーク結果の再現ではない。

## 検証結果

上記の小規模チェック（300試行、seed 7/42/123、CPU）では、greedy評価は3/3で成功し、
いずれも最短の3手。直近100試行の到達率はそれぞれ99%、97%、100%だった。
seed 42は初期Actorのgreedy評価も成功しており、この小さな迷路だけで
困難な探索問題に対する優位性を主張することはできない。
これは縮小設定での動作確認であり、論文の定量結果の再現を意味しない。

## 保存と再開

`GridWorldSGCRL.checkpoint()`はネットワーク、optimizer、適応係数、リプレイ、
途中の軌跡、専用乱数状態をJSON互換形式で保存する。
`TrainingSession.result(checkpoint=True)`と`TrainingSession.restore()`で
途中の試行から再開できる。評価用コピーにはリプレイを含めない。

## 更新速度とライブ表示

SGCRLのリプレイが既定の256MiB以内なら、状態・行動・将来ゴールを連続Tensorとして
計算デバイス上にキャッシュし、エピソード・時点・割引将来ゴールの抽出をまとめて行う。
エピソードの一様抽出と、同一エピソード内の割引幾何分布は維持する。
上限を越えたリプレイはメモリを重複確保せず、CPUでの従来の抽出へ戻る。
`crl_replay_cache_mb`で上限を変更でき、0ならキャッシュを無効化する。
CUDAではfused Adamを使用する。GridWorldの固定障害物・位置特徴も再利用する。
初期ランダム収集では使用しないActor推論と特徴のGPU転送を省く。
既定のMLPでは、全行動のQスコアを計算する最初の層を`W_s*s + W_a*a + b`に分け、
同じ状態の大きな行列積を行動数分繰り返さない。正例・シャッフルゴール間でも
状態行動表現を再利用する。独自エンコーダーを使う場合は一般の計算へ戻る。
`crl_share_actor_features=False`でこの共有を無効化できる。
損失・ネットワーク規模・既定バッチサイズ・更新回数比率は変更しない。

抽出に専用Tensor乱数生成器を導入したため、旧実装と同じseedでも経験の抽出列は変わる。
新形式のチェックポイントは抽出用乱数も保存し、同じデバイス・実装での再開を再現する。
CPU/CUDAを切り替えた場合は非互換の乱数状態だけ再設定する。

```powershell
python GridWorld/scripts/benchmark_sgcrl.py --device cuda --output output/sgcrl/update_performance_final.json
```

RTX 5070 Ti、状態800次元・ゴール800次元・隠れ層256×2・表現64・バッチ256・
リプレイ10,000件の比較では、従来の抽出＋通常Adamが121.7更新/秒、
キャッシュ＋fused Adam＋表現共有が183.4更新/秒（3回の中央値、約1.51倍）だった。
CPUの1スレッド比較では29.2更新/秒から41.9更新/秒（約1.43倍）だった。
これは合成特徴で測定した重み更新の速度で、成功率100%までの所要時間の保証ではない。

UIの学習パラメータから計算デバイスとバッチサイズを指定できる。
CUDAの離散SGCRLは最初の3更新後にCUDA Graphを生成し、同形状のバッチで再利用する。
GridWorldでは、同じマスのActor確率を重み更新までCPU側にキャッシュして行動を抽出する。
更新後はキャッシュを破棄する。行動抽出の乱数は専用CPUジェネレーターで保持し、checkpointに保存する。
従来のGPU乱数とは系列が異なるため、同じseedでも旧バージョンと軌道は一致しない。
GPUリプレイキャッシュが有効な場合は経験サンプリングもGraphに含め、経験追加時にはGraphを作り直す。
進捗の「CUDA Graph有効」で動作を確認できる。キャプチャできない場合は通常処理に戻る。
`crl_cuda_graph: false` で無効化できる。CPU、連続行動、SGCRLの派生クラスは通常処理を使う。
Graphの損失・統計値の有限性は各更新の直後に確認するため、異常時はその更新後に停止する。
リプレイキャッシュ上限も指定できる。実行表示が「CPU抽出」になっている場合は、
GPUメモリの容量に合わせて上限を増やすと、より多くの経験をGPUに保持できる。
バッチを512や1024に増やすと1回あたりのGPU処理量は増えるが、
対照学習の負例数と学習挙動も変わるため、自動では変更しない。
実行中は実際のデバイス・更新回数・今回の更新進捗・直近の更新速度を表示する。
時間表示は行動完了を待たずに現在の処理中の経過時間を取得し、
ブラウザ側でも直近の計測値から連続表示する。保存する確定時間はサーバー側で計測する。
CPUとCUDAを切り替えた復元では乱数状態の形式が異なるため、行動乱数だけを
保存seedで再初期化する。重み・optimizer・経験は保持するが、切り替え前と
同一の行動列にはならない。デバイス情報がない旧保存形式にも対応する。

SGCRL関連の数式・勾配・再開・APIテストは通過。既存のGridWorld全体テストには、
今回変更していないGRAGのcheckpoint未実装、方策検査、APIタイムアウトなどの失敗が残る。
