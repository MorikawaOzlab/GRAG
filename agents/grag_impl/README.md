# GRAG

現在の `grag.py` は作成途中です。以下のPPO・保存・目標判定の詳細は旧実装（`anget_reform.py`）の仕様も含み、現在の `grag.py` の実装状況とは異なります。

`agents/grag.py` は `GRAG` の公開入口です。関連する実装・テスト・仕様は
この `agents/grag_impl/` にまとめます。
`grag.py` が観測変換・経験収集・PPO更新を担当し、`networks.py` が
CleanRL型Actor-Critic（`Agent`）を定義します。

## 第1段階: 目標文と達成判定

確認済みの仕様:

- 目標は自然言語の文と判定用座標の組で表す。
- 判定用座標は既存の観測の `goal` を使う。座標は `[行, 列]`、左上が `[0, 0]`。
- 目標文は `GRAG(settings)` に渡す `settings["goal_text"]` で指定する。
  未指定または `None` の時は「マス [行, 列] に到達する」を生成する。
- 達成判定は `position == goal`。目標文は意味解析しないため、指定する文と座標の
  意味の一致は呼び出し側が保証する。
- 空の文、不正な座標、マップ範囲外の目標は `ValueError`。
- 目標の取得・判定は、観測やエージェントの状態を変更しない。
- 環境の報酬ルールは最終ゴール到達時に1、それ以外は0。

使用例（プロジェクトのルートで実行）:

```python
from agents.grag import GRAG
from gridworld.storage import load_settings, read_json, WORLD
from gridworld.world import GridWorld

settings = load_settings()
settings["goal_text"] = "ゴールのマスに到達する"
agent = GRAG(settings)
observation = GridWorld(read_json(WORLD)).reset()

goal = agent.goal_for(observation)
print(goal.text, goal.target)
print(agent.is_goal_reached(observation))
```

Pythonから使う時は `load_settings()` の後に `goal_text` を追加してください。

## 検証

プロジェクトのルートで:

```powershell
python -m unittest -v agents.grag_impl.tests.test_goals
```

目標入力の検証に加え、壁への衝突とゴール到達を含む実際の `GridWorld` の遷移で、
達成判定が環境の `reached_goal` と一致することを確認します。

## ニューラルネットワークによるPPO

`GRAG` の行動選択・学習は、CleanRLの離散PPOを現在のグリッドワールドに
接続した実装です。既存の `train(..., agent_type=GRAG)` と学習画面で使えます。
環境の実行・報酬・試行終了は共通実行層が担当し、AIは観測辞書だけを受け取ります。

### 依存パッケージ

Python 3.10以降とPyTorch 2.11以降が必要です。使うPython環境で次を実行します。

```powershell
python -m pip install -r agents/grag_impl/requirements.txt
```

CPU版を指定して導入する場合:

```powershell
python -m pip install -r agents/grag_impl/requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

PPOはCPUで動作します。
Gymnasium・NumPy・tyro・TensorBoard・Weights & Biasesは必要ありません。

### モデルと学習処理

- ActorとCriticはそれぞれ、隠れ層2層・tanh活性化のMLPです。
- 入力は障害物・ゴール・現在位置の3チャネルを平坦化した配列と、
  正規化した現在位置・ゴール座標の4値です。入力次元は `3 * rows * cols + 4`。
  目標文の言語埋め込みは含めません。
- 出力はActorが上下左右4行動のlogits、Criticが状態価値1値です。
  衝突する行動もサンプリング対象です。
- `choose_action()` は学習時に方策から行動をサンプリングし、選択時の
  log確率と価値を保持します。`learn()` には同じ観測・行動を渡してください。
- `learn()` は経験を収集し、指定行動数ごとに更新します。
  `end_episode()` は残った経験も更新するため、最後の試行まで学習に使います。
- GAEでadvantageと価値の教師信号を作り、PPOのクリップ付き方策損失、
  クリップ付き価値損失、エントロピー項をAdam（`eps=1e-5`）で最適化します。
  価値損失は誤差の二乗平均の1/2で、クリップ幅は方策と同じ値です。
  advantageはミニバッチごとに正規化し、1件の場合はそのまま使います。
  分散のない複数サンプルでは正規化後のadvantageを0にします。
  勾配ノルムを制限し、エポック末尾で指定KL値を超えたら次の反復を打ち切ります。
- ゴール到達では次状態の価値を0にし、最大行動回数による終了では
  最後の次状態の価値を使います。試行をまたいでGAEを伝播させません。
- `greedy=True` は最大確率の行動を返し、経験収集・乱数消費・学習を行いません。
  モデルは最初の行動選択時に初期化するため、初回の評価でも初期化は行います。
- 学習画面は1環境の継続学習で、学習率は標準では指定した値を保ちます。
  Pythonでは`ppo_anneal_lr=True`と`ppo_total_timesteps`の指定で線形減衰を使えます。
  ベクトル環境・CartPole用の実行ループは使いません。
- 同じインスタンスでは同じ行数・列数のマップを使います。
  サイズの異なるマップには新しいGRAGを生成してください。

### 設定

学習画面ではPPOの学習率・隠れ層幅・共通の割引率・シードを指定できます。
Pythonでは `load_settings()` の後に必要なキーを追加します。
Q学習の `learning_rate`・`epsilon` 等はPPOには使いません。

| キー | 標準値 | 内容 |
|---|---|---|
| `ppo_learning_rate` | 0.00025（画面では0.0003） | Adamの学習率 |
| `ppo_gamma` | `discount_factor`、未指定時0.99 | 割引率 |
| `ppo_gae_lambda` | 0.95 | GAEのlambda |
| `ppo_clip_range` | 0.2 | 方策確率比のクリップ幅 |
| `ppo_epochs` | 4 | 1回の更新でデータを反復する回数 |
| `ppo_minibatch_size` | 64 | ミニバッチの行動数 |
| `ppo_rollout_steps` | 128 | 試行途中で更新する収集行動数 |
| `ppo_hidden_size` | 64 | Actor・Criticの各隠れ層の幅 |
| `ppo_value_coef` | 0.5 | 価値損失の係数 |
| `ppo_clip_value_loss` | `True` | 旧予測値を基準に価値損失をクリップ |
| `ppo_entropy_coef` | 0.01 | エントロピーボーナスの係数 |
| `ppo_max_grad_norm` | 0.5 | 勾配ノルムの上限 |
| `ppo_normalize_advantage` | `True` | ミニバッチごとにadvantageを正規化 |
| `ppo_target_kl` | `None` | 更新打ち切りのKL閾値。`None` で無効 |
| `ppo_anneal_lr` | `False` | 指定した総ステップ数を使う線形学習率減衰 |
| `ppo_total_timesteps` | `None` | 学習率減衰の基準となる総行動数。減衰時は必須 |
| `seed` | 学習設定の値 | モデル初期化・行動選択・ミニバッチ順のシード |

### 実行と保存

プロジェクトのルートで実行する例です。

```python
from agents.grag import GRAG
from gridworld.storage import DATA, load_settings, write_json
from gridworld.training import train

map_data = {
    "name": "PPO動作確認", "rows": 3, "cols": 3,
    "start": [0, 0], "goal": [2, 2], "obstacles": [[1, 1]],
}
settings = load_settings()
settings.update({
    "episodes": 200,
    "max_steps": 40,
    "ppo_learning_rate": 0.0003,
    "ppo_hidden_size": 64,
})
result = train(map_data, settings, agent_type=GRAG)
write_json(DATA / "grag_result.json", result)
print("成功数:", result["successes"], "/", settings["episodes"])
```

`export()` のデータは、既存の結果JSONの `q_table` キーに格納されます。
内容はPPOの設定、マップサイズ、Actor・Criticの重み、学習回数、最後の更新の
損失・エントロピー・KL等です。重みはJSONで保存可能な配列に変換します。

復元して行動を選ぶ例:

```python
from agents.grag import GRAG
from gridworld.storage import DATA, read_json
from gridworld.world import GridWorld

result = read_json(DATA / "grag_result.json")
agent = GRAG.from_export(result["q_table"])
observation = GridWorld(result["map"]).reset()
print(agent.choose_action(observation, greedy=True))
```

`export()/from_export()` の復元対象はモデルと設定です。
正確な学習再開には本体の `checkpoint()/load_checkpoint()` を使います。
保存の実装は環境に依存しない `agents/ppo_checkpoint.py` です。
こちらはoptimizer・乱数状態・更新前の有限の経験バッファーも保存します。
以前のグリッドPPOの重み・Adam状態も復元できます。復元後の更新には現在の
GRAGの学習則を使います。`export()` の `implementation` は `cleanrl_grid_v1` です。

画面では「GRAG」を選び、実際の学習をアニメーションで観察できます。
好きなタイミングでN試行を高速学習し、同じAIの観察へ戻れます。
高速学習中も曲線・ヒートマップが更新され、途中停止・試行途中からの続行ができます。
PPOのCPU計算は論理CPU数−2スレッド（最低1）に設定します。
共通の学習セッションとPPOの続行用保存は
[共通実行層の仕様](../../gridworld/training_runtime/README.md)を参照してください。

### PPOの検証

```powershell
python -m unittest -v gridworld.tests.test_app agents.grag_impl.tests.test_goals agents.grag_impl.tests.test_ppo agents.grag_impl.tests.test_cleanrl_grid
```

実グリッドでの範囲外・障害物への衝突とゴール報酬、GAEの終了処理、
方策と価値のクリッピング、報酬による方策更新、
試行末尾の更新、評価時の副作用、シード再現性、モデルの保存・復元、
小さな実グリッドでの学習とゴールまでの再生を確認します。
さらに、環境オブジェクトを使わないネットワーク、価値損失の勾配、
1件のミニバッチ、学習率減衰の再開、以前の重み・Adam状態の互換性を検証します。
PyTorchがない環境では、ニューラルネット部分のテストをスキップします。

アルゴリズムの参照:
[PPO論文](https://arxiv.org/abs/1707.06347)、
[GAE論文](https://arxiv.org/abs/1506.02438)。

ヒートマップのマスをクリックすると、`action_probabilities(observation)` から取得した現在の4方向の確率を表示します。この照会はモデル・乱数・経験を変更しません。AI本体は環境・HTTP・画面を呼びません。
