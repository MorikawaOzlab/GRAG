# 共通のライブ学習セッション

`TrainingSession` は同じ学習器と現在の試行を保持し、`step()` で1行動だけ進めます。観察と高速学習は同じメソッドを呼び、表示の有無と行動間の待ち時間だけを変えます。観察中も実際に学習します。

## 画面の操作

- 「新しい学習」: 新しい学習器を学習0回・一時停止で作成。
- 「観察して学習」: 現在の位置から1行動ずつ実行・表示。
- 「N試行を高速学習」: アニメーションを省略して、完了試行数をN増やす。
- 「一時停止」「高速学習を停止」: 現在の1行動の学習処理が終わったら停止。
- 「観察速度」: 観察時の行動間隔を変更。
- 「1試行の最大行動回数」: 次の試行から新しい上限を適用。

観察中にN試行を高速学習すると、完了後は観察へ戻ります。一時停止から高速学習を開始した場合は、一時停止に戻ります。進行中の試行も最初の完了試行として数えます。途中停止でモデル・試行・乱数・集計はリセットしません。

## ヒートマップと行動確率

試行錯誤を観察とヒートマップを横並びにし、その下に学習曲線を全幅で表示します。
画面幅が800px以下の場合は観察、ヒートマップ、学習曲線の順に縦並びになります。
ヒートマップの表示倍率は50〜250%で変更できます。
大きなマップでも1マスを最低44pxで表示し、スクロールして確認できます。

通行可能なマスをクリックすると、その位置にAIがいると仮定した現在の行動確率を表示します。
ポップアップは取得時点の完了試行数・総行動数・訪問数を表示し、
「現在の確率に更新」で再取得します。学習を自動停止したり、位置を移動したりはしません。
確率は学習時の選択確率で、Q学習では現在の探索率も含みます。
衝突する方向にも確率がある場合は「衝突」を付けて表示します。
障害物は確率表示の対象外、ゴールでは仮想的な方策の出力と試行終了の注意を表示します。

`POST /api/train/policy`:

```json
{"session_id": "<ID>", "mode": "ppo", "map_name": "例", "position": [3, 4]}
```

主な応答は `probabilities`（4方向の確率）、`blocked_actions`、
`completed_episodes`、`total_steps`、`visits`、`supported`、`wall`、`goal`。
セッションID・AI・マップの一致と座標を検証します。
学習中は1行動の学習処理の間で取得し、更新途中のモデルを読みません。
保存済みセッションの読み取りだけで、新しいジョブ・保存ファイル・常駐モデルを増やしません。

## AIと環境の分離

AI本体は観測・学習パラメータを受け取り、行動・確率・保存データを返します。
`GridWorld`、HTTP、画面・ファイル保存は呼びません。
環境側は公開メソッドを使い、AIのモデル・optimizer・乱数・経験バッファーの内部属性へ触れません。

確率表示に対応するAIは、任意の公開メソッド
`action_probabilities(observation) -> dict[str, float]` を実装します。
返すキーは `up`, `down`, `left`, `right`、各値は0〜1で合計1。
観測・モデル・乱数・学習バッファーを変更しないことが契約です。
未対応のAIはメソッドを省略するか、`NotImplementedError` を返します。

PPOの学習保存はAI側の `agents/ppo_checkpoint.py` に置き、
GRAGとPPOAgentが `checkpoint()/load_checkpoint()` を公開します。
`ppo_adapter.py` は画面用の名前を付けるだけです。

AIへ渡す学習設定は環境側とは別のコピーにします。試行ごとの上限をAIへ通知する場合は、任意の公開メソッド `set_episode_limit(max_steps)` を呼び、AIの設定属性を環境側から書き換えません。

## 呼び出し順

```python
from gridworld/training_runtime import TrainingSession

session = TrainingSession(map_data, settings, agent_type=YourAgent)
live = session.step()  # 実際の行動・遷移・学習を1回行う
session.run_batch(100)  # 同じ試行の続きから100試行を完了する
result = session.result(checkpoint=True)
restored = TrainingSession.restore(result, agent_type=YourAgent)
restored.step()
```

`step()` の処理は次の順です。

```text
新しい試行なら gridworld.world.reset()
agent.choose_action(observation)
gridworld.world.step(action)
報酬を計算（到達1、それ以外0）
agent.learn(observation, action, reward, next_observation, reached_goal)
訪問数・総行動数・現在位置を更新
到達または行動上限なら agent.end_episode() → 試行集計を更新
現在の状態1件を返す
```

衝突も1行動です。開始時の配置と衝突を訪問数には加算しません。
学習器は `choose_action`, `learn`, `end_episode`, `export` を実装します。
再起動後に復元する学習器は `checkpoint()` と `load_checkpoint(state)` も必要です。

## ジョブとAPI

`gridworld.training_app.LEARNING_AGENTS` にQ学習、`PPOAgentSession`（独立したPPOエージェント）、`GRAGSession`（GRAGのPPO）を登録しています。
`ThreadPoolExecutor(max_workers=1)` で同じ学習器を操作します。

`POST /api/train` の例:

```json
{
  "map_name": "例",
  "mode": "ppo",
  "execution": "paused",
  "training_settings": {"max_steps": 200, "ppo_hidden_size": 64}
}
```

`mode="ppo"` は独立コピーのPPO、`mode="grag"` はGRAGです。
応答の `session_id` を次の操作に渡します。

```json
{"action": "observe", "session_id": "<ID>"}
```

```json
{"action": "skip", "episodes": 100, "session_id": "<ID>"}
```

上記操作の送信先は `POST /api/train/control` です。
`action` は `pause`, `observe`, `skip`, `speed`, `limit`。
`speed` に `step_delay_ms`、`limit` に `max_steps` を指定します。
高速学習回数は1〜1,000,000、観察間隔は20〜2,000ミリ秒です。

`GET /api/train/status` の `preview` から、曲線・訪問数・現在位置・累計値を取得します。
高速学習中は約0.1秒間隔で公開し、ブラウザは実行中150ミリ秒、一時停止中250ミリ秒の間隔で取得します。
PPOの1回の更新に時間がかかる場合、その処理の完了後に公開・停止します。
別の学習を開始するには、現在の学習を一時停止します。

PPOの計算スレッド数は `max(1, (os.cpu_count() or 1) - 2)` です。
学習ワーカー自身と、チェックポイントを復元するスレッドで適用します。
ステータスの `cpu_threads` は学習ワーカー内で取得した実効値です。
観察と高速学習の間でスレッド数を切り替えません。

## メモリとチェックポイント

各試行の動画・座標列は記録しません。`live` は現在の状態1件です。
曲線は最大2,000点、直近の平均用の報酬・行動数は各100件に制限します。
画面へは最大約700点に間引いた曲線を渡し、モデル・optimizerは送りません。
累計試行数・行動数・成功数・訪問数は正確な値を保ちます。

一時停止、高速学習の完了、新しい実験への切り替え時に
`data/training_sessions/<ID>.json` を上書き保存します。
現在の実験の保存結果は `data/latest_result.json` にも保存します。
途中の位置、行動数、報酬、現在の試行の上限も保存し、途中から復元できます。
未保存の実行中にプロセスを強制終了した場合は最後の保存時点へ戻ります。

Q学習はQ表・探索率・乱数状態を保存します。
PPO本体の `checkpoint()` は、環境に依存しない `agents/ppo_checkpoint.py` を使い、モデル・Adam・専用乱数生成器・収集中の有限の経験バッファーを保存します。
`PPOAgent.export()/from_export()` と `GRAG.export()/from_export()` は推論用モデル保存として維持し、
学習を正確に続行するときはセッションのチェックポイントを使います。

## 互換用の実行

`execution="batch"` は指定回数の学習と学習後の方策評価を行う従来APIです。
`POST /api/train/evaluate` と `TrainingSession.evaluate()` も維持しています。
評価は学習器のコピーで `greedy=True` の行動を選び、元のモデル・乱数・集計を変えません。
これらの明示的な評価だけが、1回の評価走行の `frames` を生成します。
ライブ画面の観察・高速学習は評価やフレーム生成を呼びません。

`gridworld/parallel_training.py` の並列Q学習も独立したスクリプト用処理で、ライブ学習には使いません。

## 検証

```powershell
python -m unittest -v gridworld.training_runtime.tests.test_session gridworld.training_runtime.tests.test_api gridworld.training_runtime.tests.test_ppo_session gridworld.training_runtime.tests.test_live_training gridworld.training_runtime.tests.test_policy_inspection
```

同じ学習器の継続、途中停止・復元、N試行の正確な完了、観察への復帰、
高速学習中の曲線・ヒートマップ更新、履歴の上限、PPOのoptimizer・乱数・経験の復元を確認します。
