# GRAG

GRAGを開発・検証するための障害物付き2次元グリッドワールドです。

詳細仕様: [AI実行環境仕様書](AI_RUNTIME_SPEC.md)  
学習の実装: [共通学習セッション](gridworld/training_runtime/README.md)

Python 3.10以降が必要です。環境・Webアプリ・Q学習は標準ライブラリで動作します。PPOエージェントとGRAGの対照学習にはPyTorchが必要です（[導入・検証](agents/grag_impl/README.md#保存検証)）。

## フォルダ構成

```text
LLM_Agentic_RL/
├─ gridworld/              # 環境・Webアプリ・学習の実行処理
│  ├─ world.py
│  ├─ map_app.py
│  ├─ settings_app.py
│  ├─ training_app.py
│  ├─ storage.py
│  ├─ training_runtime/
│  ├─ templates/
│  ├─ static/
│  └─ tests/
├─ agents/                 # AIの実装
│  ├─ grag.py              # GRAGの公開入口
│  ├─ grag_impl/           # 作成中のGRAG・旧実装・テスト
│  │  ├─ grag.py           # GRAGクラスの実装
│  │  └─ grag_interface.py # インターフェースひな形
│  ├─ ppo_agent.py
│  ├─ ppo_agent_impl/
│  ├─ q_learning.py
│  └─ ppo_checkpoint.py
├─ scripts/                 # 3アプリの起動用バッチファイル
└─ data/                   # 保存済みマップ・設定・学習結果
```

GRAGの編集先は `agents/grag_impl/grag.py` です。既存の参照用実装は `agents/grag_impl/anget_reform.py` にあります。
アプリはプロジェクトのルートから `python -m gridworld.<モジュール名>` で起動します。

## 起動

次のファイルをダブルクリックすると、各アプリを起動できます。

- `scripts/map_app.bat`：マップ作成
- `scripts/settings_app.bat`：環境のルール
- `scripts/training_app.bat`：AI学習

表示されたURLをブラウザで開いてください。終了するときは、そのアプリのターミナルで `Ctrl+C` を押します。

PowerShellから起動する場合:

```powershell
.\scripts\map_app.bat
.\scripts\settings_app.bat
.\scripts\training_app.bat
```

スクリプトはプロジェクトの `venv/Scripts/python.exe` を優先して使います。別のPythonを使う場合は、環境変数 `GRAG_PYTHON` に実行ファイルのパスを指定してください。`venv` がない場合は `python` を使います。

モジュールとして直接起動する場合は、プロジェクトのルートから `python -m gridworld.training_app` などを実行します。

| URL | 用途 |
|---|---|
| http://127.0.0.1:5001 | マップの作成・保存 |
| http://127.0.0.1:5002 | 環境のルールの確認 |
| http://127.0.0.1:5003 | AI学習・観察・高速学習 |

各アプリは単独で起動できます。学習には保存済みマップが必要です。
環境のルール画面は上下左右の移動、衝突、報酬、座標を説明します。試行数・最大行動回数・学習パラメータはAI学習画面で指定します。

このPCで動作確認したPythonは次の実行ファイルです。PPOを使う場合は、PyTorchが入っているPythonで起動してください。

```powershell
& "C:\Users\2kame\.pyenv\pyenv-win\versions\3.11.9\python.exe" -m gridworld.training_app
```

コードを更新した後は、起動済みの学習サーバーを再起動し、ブラウザを更新してください。

## 学習を観察する

1. マップ・AI・1試行の最大行動回数を選びます。学習器はQ学習、PPOエージェント、GRAGです。
2. 「新しい学習」で学習0回の状態を作ります。
3. 「観察して学習」で、実際に行動を選び、失敗し、学習している現在のAIを表示します。
4. 好きなタイミングでN試行の高速学習を実行できます。行動表示を省略し、同じAIを学習させます。
5. 観察中に高速学習を始めた場合、N試行完了後に観察へ戻ります。一時停止から始めた場合は完了後も一時停止します。
6. 「一時停止」または「高速学習を停止」で途中停止できます。再開時は同じAI・同じ試行の続きです。

高速学習中も学習曲線とヒートマップが更新されます。途中の試行を進めている場合、その試行の完了を高速学習の1試行目として数えます。停止要求は実行中の1行動の学習処理が終わった時点で反映されます。

試行錯誤を観察とヒートマップを横並びにし、その下に学習曲線を全幅で表示します。画面幅が800px以下の場合は縦並びになります。表示倍率を50〜250%に変更でき、大きなマップはスクロールできます。曲線は直近100試行の平均報酬、ヒートマップは累計訪問数です。完了試行数、直近100試行の成功率・平均行動数、総行動数も表示します。

最大行動回数の変更は次の試行から適用します。シード、割引率、Q学習の探索率、PPOの学習率・隠れ層幅の変更は「新しい学習」で適用します。

ヒートマップのマスをクリックすると、取得時点の上下左右の行動確率がポップアップで表示されます。「現在の確率に更新」で最新の状態を確認できます。Q学習では探索率も含む確率です。確認では学習・乱数・現在位置を変更しません。

## 記録とCPU

学習アニメーション用の試行録画・全行動履歴は保存しません。保持するのは現在位置、学習器、訪問数、集計値です。曲線の履歴は最大2,000点に間引き、画面へは最大約700点を配信します。直近100試行の平均と累計値は間引かず計算します。

一時停止や高速学習の完了時に、実験ごとの最新状態を上書き保存します。Q学習はQ表・探索率・乱数状態、PPOはモデル・Adam・乱数状態・更新前の経験バッファーを保存し、途中の試行から再開できます。PPOの経験バッファーは学習に必要な有限のデータで、標準では128行動ごとに更新します。

学習ワーカーは1つです。PPOの内部計算のCPUスレッド数は、観察・高速学習ともに `max(1, 論理CPU数 - 2)` に設定します。このPCは論理CPUが20なので18です。複数のAIを裏で同時に学習させる処理はありません。

## 保存ファイル

| パス | 内容 |
|---|---|
| `data/maps/*.json` | 保存済みマップ |
| `data/training_settings.json` | 新しい学習で指定した学習設定 |
| `data/settings.json` | 旧設定。新しい学習設定がない場合に読み込む |
| `data/gridworld.world.json` | 選択マップのスナップショット |
| `data/latest_result.json` | 最新の保存済み学習状態・集計 |
| `data/training_sessions/<ID>.json` | 実験ごとの最新チェックポイント |

同じ実験のチェックポイントは上書きし、試行ごとのファイルは作りません。新しい実験を作っても以前の実験ファイルは残ります。別のマップ・AIへの続行はできません。

マップJSONは `name`, `rows`, `cols`, `start: [行, 列]`, `goal: [行, 列]`, `obstacles: [[行, 列], ...]` です。到達可能性は検査しません。報酬はゴール到達時1、それ以外0です。

## 独自AI

AI本体は環境・HTTP・画面に依存せず、観測を受け取って行動や確率を返します。PPOのモデル・optimizer・乱数・経験の保存もAI側の `agents/ppo_checkpoint.py` が担当します。環境側は公開メソッドだけを呼びます。確率表示用の公開メソッドは `action_probabilities(observation)` です。

[PPOエージェント](agents/ppo_agent_impl/README.md) はコピー時点のGRAGのPPOを独立させた実装です。現在の[GRAG](agents/grag_impl/README.md)は「A Single Goal Is All You Need」の学習則を離散迷路へ適用したSGCRLです。独立PPOの公開入口は `agents/ppo_agent.py`、本体は `agents/ppo_agent_impl/agent.py` です。

`choose_action`, `learn`, `end_episode`, `export` を持つクラスを `gridworld.training_app.LEARNING_AGENTS` に登録すると、共通の観察・高速学習を使えます。再起動後の続行には `checkpoint` と `load_checkpoint` も実装します。

従来の `gridworld.training.train(...)` と `gridworld.parallel_training.train_parallel(...)` はスクリプト用に残しています。並列Q学習は独立した学習器のQ表を平均する方式で、観察画面では使いません。

## 検証

環境・保存API・共通学習処理の基本テスト（17件）:

```powershell
python -m unittest -v gridworld.tests.test_app gridworld.training_runtime.tests.test_session
```

現在のGRAGと共通実行層のテスト:

```powershell
python -m unittest -v gridworld.tests.test_app agents.grag_impl.tests.test_grag_update agents.grag_impl.tests.test_training_settings gridworld.training_runtime.tests.test_session gridworld.training_runtime.tests.test_api gridworld.training_runtime.tests.test_ppo_session gridworld.training_runtime.tests.test_live_training gridworld.training_runtime.tests.test_policy_inspection
```

実際の学習ステップ、観察と高速学習の切り替え、進捗の更新、停止・途中再開、保存・復元、曲線のメモリ上限、PPOの実効CPUスレッド数を検証します。

GRAGの旧PPO仕様を前提とする `test_goals.py`、`test_ppo.py`、`test_cleanrl_grid.py` は現行のSGCRLには適用しません。

GRAGの公開クラスは `from agents.grag import GRAG` で読み込みます。画面・APIの識別名は `grag` です。旧識別名 `myagent_ppo` のAPI入力・保存結果は、読み込み時に `grag` として扱います。
