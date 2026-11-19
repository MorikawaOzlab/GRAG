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
設定変更は新しい学習に適用されます。1更新あたりの既定の収集数は65,536行動です。公開版の128行動×512環境に対応します。更新は指定行動数に達した時点で行うため、試行の途中でも発生します。
一時停止時には未更新の軌跡・モデル・Adam・乱数状態も保存し、アプリ再起動後に続行できます。
評価と確率表示は学習用の乱数や軌跡を変更しません。
評価時は公開版の既定設定と同じく、方策の確率に従って行動を抽出します。`cppo_evaluation_greedy` を有効にすると、確率が最大の行動を選びます。

## 公開実装との対応

移植元は公開コードの [Navix向けCPPO](../agents/CPPO/mava/systems/icrl/anakin/ff_ppo_crl_nx.py) です。

- 方策と状態・行動エンコーダーは、それぞれ3層のCNNを持ちます。観測は壁・目標・現在位置の3チャンネルです。大きなマップでは進行方向を持たないGridWorldに合わせ、北を上に固定した最大7×7の近傍を使います。行動は上下左右、目標と到達位置は行・列の座標です。
- GridWorldの1環境から、同じ方策で固定長の区間を逐次収集します。ゴール到達や時間制限で試行を区切り、各区間内で同一試行の将来到達位置を割引付きで抽出します。公開版と同じく、その遷移の到達位置にも微小な重みを与え、各区間の末尾は学習サンプルから除きます。短い収集設定でミニバッチ1個未満の場合は、使える行を1バッチにまとめます。
- 対照学習にはミニバッチの全ペアによるforward InfoNCEとlogsumexp正則化を使います。公開版と同じく、負のL2距離によるQ値と全4行動の方策期待値からadvantageを計算します。シャッフル前に学習に使う全サンプルで一度だけ正規化して固定し、方策をクリップ付きPPOとentropy項で更新します。
- 方策とエンコーダーは別々のAdamで更新し、それぞれ勾配ノルムを制限します。各学習率はミニバッチ更新ごとにcosineで減衰します。既定値はNavixのYAML設定を参照し、収集65,536行動、区間長128、バッチ512、学習率0.00025、entropy係数0.01を使用します。公開版の個別実験スクリプトには一部の値を上書きする設定があります。

APIの `training_settings` では `cppo_` に続けて `hidden_size`, `depth`, `actor_hidden_size`, `repr_dim`, `rollout_steps`, `rollout_length`, `batch_size`, `epochs`, `total_updates`, `learning_rate`, `critic_learning_rate`, `gamma`, `clip_coef`, `entropy_coef`, `max_grad_norm`, `lr_end`, `logsumexp_penalty_coef`, `normalize_advantage` を指定できます。`critic_learning_rate` の省略時は `cppo_learning_rate` と同じ値を使います。`normalize_advantage` は既定で有効です。保存済みのセッションで明示的に無効化されている場合は、その設定を引き継ぎます。
公開版と同じ選択肢として、`cppo_energy_fn` に `norm`, `l2`, `dot`, `cosine`、`cppo_contrastive_loss_fn` に `fwd_infonce`, `bwd_infonce`, `sym_infonce`, `binary_nce`、`cppo_lr_decay_type` に `cosine`, `linear`, `none` を指定できます。各リストの先頭が既定値です。
通常の `discount_factor`、`learning_rate`、epsilonの設定はCPPOには使いません。画面の報酬曲線は到達率の評価用です。

## 小規模な動作確認

```powershell
.\venv\Scripts\python.exe GridWorld/scripts/verify_cppo.py --episodes 3000 --seeds 7 42 123 --rollout-steps 128 --output output/cppo/default_validation.json
```

標準の検証迷路は2×3、壁1個、最短経路3行動、試行上限16行動です。
`--map <JSON>` で保存済みマップを指定でき、`--hidden-size`、`--depth`、`--rollout-steps`、`--max-steps` も変更できます。
JSONにはマップ、指定設定、シード、学習曲線、最終経路、更新回数、損失を保存します。

## テスト

CPPOの保存形式v3は学習状態をバイナリで保持し、JSONにはbase64で格納します。
モデルを`q_table`とチェックポイントに二重保存せず、最新結果ファイルは表示用集計だけにします。
実験ごとのファイルにはoptimizer・乱数・途中の経験も保存するため、再起動後も続行できます。
旧v1/v2はネットワーク構成が異なるため、CNN版では新しい学習を開始してください。

```powershell
$env:PYTHONPATH = "$PWD\GridWorld;$PWD"
.\venv\Scripts\python.exe -m unittest gridworld.training_runtime.tests.test_cppo_session gridworld.training_runtime.tests.test_api.SessionApiTests.test_cppo_train_policy_evaluate_and_restart -v
```

報酬非依存、Actorとエンコーダーの更新、advantageの全行動期待値、HERの試行境界、JSON経由の途中復元、評価時の不変性、HTTP経由の学習・確率表示・評価・復元を確認します。
