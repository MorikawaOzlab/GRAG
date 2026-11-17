# GridWorldでCPPOを実行する

実装は [`agents/cppo.py`](../agents/cppo.py) の `CPPOAgent` に集約しています。
他のエージェントを継承・参照せず、PyTorch以外のライブラリを必要としません。

## Web UI

リポジトリのルートで実行します。既存の `venv` にPyTorchがあれば追加インストールは不要です。

```powershell
.\GridWorld\scripts\training_app.bat
```

1. http://127.0.0.1:5003 を開き、保存済みのマップを選択します。
2. AIを **CPPO** にして「新しい学習」を押します。
3. 「観察して学習」または試行数を指定して「高速学習」を押します。
4. ヒートマップのマスをクリックすると行動確率を確認できます。

学習パラメータから学習率、隠れ層幅、将来ゴール抽出の割引率、合計収集行動数、ミニバッチサイズ、更新エポック数、entropy係数、advantage正規化の有無を変更できます。
設定変更は新しい学習に適用されます。更新は最小収集行動数に達し、2試行以上が終了した時点で行います。
既定の合計収集数は65,536行動です。論文の128行動×512環境に対応し、最大200行動の試行なら、最初の更新まで約328試行以上かかります。
旧版の合計128行動は1環境あたりの値との取り違えでした。保存済み実験の設定は維持されるため、古い実験を続行しても自動修正されません。
一時停止時には未更新の軌跡・モデル・Adam・乱数状態も保存し、アプリ再起動後に続行できます。
評価と確率表示は学習用の乱数や軌跡を変更しません。

## 原論文との対応と変更点

対象は [Self-Supervised On-Policy Reinforcement Learning via Contrastive Proximal Policy Optimisation](https://arxiv.org/html/2605.13554v1) のCPPOです。
将来到達位置によるforward InfoNCE、負のL2距離によるQ値、全4行動の方策確率で重み付けしたbaseline、PPOのclip目的関数を実装しています。
advantageはAlgorithm 1の順に、更新前のモデルで計算して固定します。環境報酬・GAE・価値ヘッド・リプレイバッファ・ターゲットネットワークは使いません。

この実装は**既存GridWorldへの移植**です。論文のベンチマークや報告スコアをそのまま再現したものではありません。

- 観測は全マップの障害物と現在位置のone-hot、目標はマップのGの座標です。視野・行動数はNavixと異なります。
- 512並列環境の代わりに、固定した同じ方策で合計65,536遷移以上になるまで完了エピソードを逐次収集します。更新後は収集データを破棄します。逐次収集と並列環境の軌跡分布は同一とは限りません。
- HERは同一試行内の次状態以降から、重み `gamma ** offset` で抽出します。時間制限でも試行を分割します。
- 負例は同じ収集バッチ内の別試行から取ります。同一試行の負例はマスクし、ミニバッチが単一試行なら別試行の負例を1件追加します。同じ座標を持つ別試行の負例は残します。
- エンコーダーと方策は独立したMLPです。CPU上で動作します。対照損失と方策損失を同じAdamの1ステップで更新します。
- 既定の4層・幅512・LayerNorm・SiLU・表現64次元、学習率0.00025、clip 0.2、1エポック、勾配上限0.5、cosine減衰を使用します。
- entropy係数0.01とミニバッチごとのadvantage正規化はこの移植での選択です。cosine減衰は1250更新で学習率1e-7に達し、その後固定します。

APIの `training_settings` では `cppo_` に続けて `hidden_size`, `depth`, `repr_dim`, `rollout_steps`, `batch_size`, `epochs`, `total_updates`, `learning_rate`, `gamma`, `clip_coef`, `entropy_coef`, `max_grad_norm`, `lr_end`, `normalize_advantage` を指定できます。`normalize_advantage` はJSONの真偽値です。
通常の `discount_factor`、`learning_rate`、epsilonの設定はCPPOには使いません。画面の報酬曲線は到達率の評価用です。

## 旧128遷移設定で行った小規模実験

```powershell
.\venv\Scripts\python.exe GridWorld/scripts/verify_cppo.py --episodes 3000 --seeds 7 42 123 --rollout-steps 128 --output output/cppo/default_validation.json
```

標準の検証迷路は2×3、壁1個、最短経路3行動、試行上限16行動です。
`--map <JSON>` で保存済みマップを指定でき、`--hidden-size`、`--depth`、`--rollout-steps`、`--max-steps` も変更できます。
JSONにはマップ、指定設定、シード、学習曲線、最終経路、更新回数、損失を保存します。

2026-11-16の旧設定での実行では、幅512・4層で3シードすべてが学習前のgreedy評価では未到達、3000試行後は3行動で到達しました。修正後の既定収集量の結果ではありません。
直近100試行の到達率はすべて100%でした。これはこの小迷路での動作検証で、大きな迷路の性能保証ではありません。
検証用の幅64・2層では1000試行後も3シードすべてgreedy評価で未到達だったため、ネットワークを縮小した設定は既定値として採用していません。

8×10サンプルマップでの方策崩壊の調査は [CPPO_INVESTIGATION.md](CPPO_INVESTIGATION.md) を参照してください。収集量の訂正と学習性能の解決は区別しています。

## テスト

CPPOの保存形式v2はTensorをバイナリで保持し、JSONにはbase64で格納します。
モデルを`q_table`とチェックポイントに二重保存せず、最新結果ファイルは表示用集計だけにします。
実験ごとのファイルにはoptimizer・乱数・途中の経験も保存するため、再起動後も続行できます。
旧v1の数値配列JSONも読み込み可能です。学習と保存はWeb UIとは別プロセスで実行します。

```powershell
$env:PYTHONPATH = "$PWD\GridWorld;$PWD"
.\venv\Scripts\python.exe -m unittest gridworld.training_runtime.tests.test_cppo_session gridworld.training_runtime.tests.test_api.SessionApiTests.test_cppo_train_policy_evaluate_and_restart -v
```

報酬非依存、Actorとエンコーダーの更新、advantageの全行動期待値、HERの試行境界、JSON経由の途中復元、評価時の不変性、HTTP経由の学習・確率表示・評価・復元を確認します。

初回の確認では、CPPO専用6件・CPPOのHTTP API 1件・共通セッション9件の計16件が成功しました。収集量の訂正後の検証は調査記録を参照してください。
共通API・既存PPOのテストも含めた27件では、GRAGSessionのcheckpoint未実装に関する4件がエラーでした。CPPOの追加ではGRAG本体を変更していません。
起動したサーバーの `/api/agents` とHTMLはHTTP 200で、CPPOの選択肢・設定項目の配信を確認しています。
ブラウザ接続ツールが利用不可で、Chromeの直接操作もツール側で未承認と判定されたため、画面の目視確認は未実施です。
