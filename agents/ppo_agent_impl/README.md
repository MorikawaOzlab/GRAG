# PPOエージェント

`agents/ppo_agent.py` が公開入口、`agents/ppo_agent_impl/agent.py` がニューラルPPO本体です。
2026-10-14時点のGRAGのPPOをコピーしています。
GRAGを継承せず、モデル・GAE・PPO更新・入力変換・目標ヘルパーを別のファイルに保持します。
今後のGRAGの変更が、このPPOの行動選択や学習則へ自動的に反映されることはありません。

学習アプリの「AI」から **PPOエージェント** を選びます。
観察・N試行の高速学習・途中停止・保存からの続行を利用できます。
「GRAG」も別の選択肢として残しています。APIのモードはそれぞれ `ppo` と `grag` です。

## 依存関係

PyTorchを使うPython環境で起動します。導入が必要な場合:

```powershell
python -m pip install -r agents/ppo_agent_impl/requirements.txt
```

モデル・設定の初期値はコピー時点のGRAGと同じです。
現在のGRAGはCleanRLの価値クリッピングとAdamの`eps=1e-5`を使います。
この独立PPOはコピー時点の価値の二乗誤差とAdamの設定を維持するため、
同じシードでも現在のGRAGと同じ学習結果になるとは限りません。
Actor・Criticはそれぞれ2隠れ層のtanh MLP、学習則はGAEとPPO-Clipです。
入力はマップの障害物・ゴール・現在位置と正規化座標で、自然言語の埋め込みはありません。

## Pythonから利用する

```python
from agents.ppo_agent import PPOAgent
from gridworld.storage import load_settings
from gridworld.training import train

settings = load_settings()
settings.update({"episodes": 100, "max_steps": 200})
result = train(map_data, settings, agent_type=PPOAgent)
```

途中保存・正確な学習再開には、環境に依存しない本体の `checkpoint()/load_checkpoint()` を使います。セッションと現在位置を合わせて保存する例:

```python
from gridworld/training_runtime import TrainingSession
from agents.ppo_agent import PPOAgent

session = TrainingSession(map_data, settings, agent_type=PPOAgent)
session.step()
saved = session.result(checkpoint=True)
restored = TrainingSession.restore(saved, agent_type=PPOAgent)
restored.step()
```

本体の `export()/from_export()` は推論用モデル保存です。
本体の `checkpoint()/load_checkpoint()` はモデル・Adam・乱数・収集中の経験を保存します。
保存処理だけをAI側の `agents/ppo_checkpoint.py` で共通化し、復元する経験の型はそれぞれの実装を使います。
環境は公開メソッドだけを使い、モデル・optimizer・経験の内部属性には触れません。
学習アプリは別のAIのセッションIDへの切り替えを拒否するため、GRAGとPPOエージェントの実験は別々に扱います。

WebでのCPU計算スレッド数は観察・高速学習ともに論理CPU数−2（最低1）。
このPCでは18です。詳しい設定・操作は [共通学習セッション](../../gridworld/training_runtime/README.md)、
共通のPPO設定は [GRAGのPPO仕様](../grag_impl/README.md#設定) を参照してください。
`ppo_clip_value_loss`・`ppo_anneal_lr`・`ppo_total_timesteps` はGRAG専用で、
この独立PPOには適用しません。

## 検証

```powershell
python -m unittest -v gridworld.training_runtime.tests.test_ppo_session gridworld.training_runtime.tests.test_live_training
```

同じシードと条件でこの独立PPOの学習結果を再現できること、
GRAGの行動選択を変更しても独立して動くこと、
モデル・optimizer・乱数・試行途中の経験の復元、Webの選択・観察・高速学習・停止、
実際の学習スレッドのCPU上限を確認します。

ヒートマップのマスをクリックすると、`action_probabilities(observation)` から取得した現在の4方向の確率を表示します。この照会はモデル・乱数・経験を変更しません。AI本体は環境・HTTP・画面を呼びません。
