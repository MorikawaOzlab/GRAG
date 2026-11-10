# BabyAI 実験アプリ

起動: ルートで `python -m babyai.app`。URL: http://127.0.0.1:5010 。
依存: `python -m pip install -r babyai/requirements.txt`。
Ollamaは `ollama pull gpt-oss:20b` の後、ローカルサーバーを起動してください。

## 操作

環境IDを指定し、PPO / Ollama / GRAG を選び「新しい実験」を押します。
1ステップ、1試行の観察、高速実行、一時停止、保存・再開、曲線、CSV出力を提供します。
Ollamaは学習を行わず各ステップで推論する比較条件です。
一時停止は実行中の推論・学習ステップが完了した後に反映されます。
保存先は `babyai/data/`。チェックポイントはローカルPythonオブジェクトを含むため、
外部から取得したpickleファイルは配置しないでください。

## Webから指定する自然言語タスク

空欄では公式BabyAIの命令と判定を使います。
英語の `go to the red ball`、`pick up the red key`、`open the blue door` は
簡易解析します。それ以外の日本語を含む任意の命令には判定条件JSONを指定できます。
命令はPPOとOllama、GRAGに同じ部分観測とともに渡されます。
PPOの命令長はUTF-8で512バイト以内です。PPOは部分観測のMLPによる基準実装です。
全体図は観察専用で、エージェントには渡しません。

例: 「赤い鍵を拾い、その後に青いドアを開ける」

```json
{"type":"sequence","goals":[
  {"type":"pickup","object":{"type":"key","color":"red"}},
  {"type":"open","object":{"type":"door","color":"blue"}}
]}
```

判定は `task.py` の `evaluate_goal` / `evaluate_task` / `parse_goal` を編集します。
`goto`, `pickup`, `open`, `putnext`, `and`, `sequence` に対応します。
`putnext` は `object` と `target` に対象条件を指定します。
座標条件 `position:[x,y]` も指定できます。
指定した対象が存在する環境IDを選んでください。命令の入力でマップは変更されません。
独自条件では達成報酬1・その他0を使い、標準命令の終了判定は使いません。
条件は各ステップ後に評価します。必要に応じて履歴・達成イベントの定義を編集してください。

## GRAG

既存のGRAGコードには変更を加えていません。
BabyAI対応版を実装したら、Webの接続先に `your_module:GRAG` を指定します。
インターフェースは `grag_adapter.py` に記載しています。
現行の座標4行動版は、そのままでは実行できないため接続時にエラーを表示します。

## 確認

`python -m unittest babyai.test_runtime -v`
PPO更新・保存復元、複合タスクの判定を検証します。
実環境の起動にはMiniGridの導入が必要です。


## Parallel PPO and Ollama output

The CPU worker setting applies to PPO fast training (new experiments).
0 selects up to four workers, leaving two logical CPUs free when possible.
1 keeps serial collection. Observation always uses one environment.
Workers collect complete episodes with a frozen shared policy. The parent
performs PPO updates from those on-policy samples after each batch.
Pause waits for the in-flight batch to finish. The observation diagram shows
the final worker state after each batch; it is not a live view of every worker.
GRAG is left unchanged; its arbitrary interface is not distributed across workers.

Ollama streams message.thinking and message.content into the output panel.
The selected action is validated before execution. Invalid output is shown as
an error and does not move the agent. All executed decisions and failed inference logs are saved
with the session. The UI retrieves older entries incrementally and keeps them
visible across decisions. Output follows the bottom until the reader scrolls up;
scrolling back to the bottom resumes following. Japanese explanations are
requested, with a separate ACTION: identifier for machine execution. The model's default thinking setting is used. Output is shown
only when the Ollama server returns it; no substitute reasoning is fabricated.


## LLM movement feedback

The Ollama prompt explains blocking walls/objects/doors and in-place turns.
It includes the front cell decoded from the partial observation, not the full map.
Observations contain last_transition (moved, blocked_forward, turned,
inventory_changed), measured from the actual transition. Ollama retains the
last eight transitions within an episode; this memory resets at episode end.
This is context memory, not LLM parameter training. PPO currently ignores these
extra dictionary fields in its encoder. Account for this input preprocessing
when interpreting comparisons. No collision-avoidance policy overrides LLM actions.


## Direction and egocentric coordinates

LLM input uses a decoded row-major view instead of the raw x-major numeric array.
The top of this view is always forward, not north. Explicit orientation fields
map forward/right/left/backward to compass names for the current heading.
Objects have relative forward/right offsets and compass displacement offsets
(east-positive, south-positive), never privileged world coordinates.
Carried objects at the agent pixel are excluded from visible floor objects.
All four headings are checked against MiniGrid get_view_coords in tests.
The observation panel also displays the current compass mapping.
