# GRAG

# 問題アダプター

基本の `GRAG` は、ネットワークの初期化時に問題アダプターを受け取れます。
GridWorld では次のようにマップからアダプターを作成します。

```python
from agents.grag import GRAG
from agents.grag_impl.gridworld_adapter import GridWorldAdapter

agent = GRAG({"seed": 1})
agent.initialize_networks(problem=GridWorldAdapter(map_data))
action = agent.choose_action(observation)
```

アダプターは `network_dimensions(action_dim)` で状態・ゴール・行動の
次元を返し、`encode_observation(observation)` で観測を数値テンソルに変換します。
GridWorld の状態は障害物・ゴール・現在位置の3チャネルです。
従来の `initialize_for_gridworld(map_data)` と、次元を直接指定する
`initialize_networks(state_dim, goal_dim, action_dim)` も利用できます。

GRAG本体の `_prepare_state` は、アダプターによる変換後に、問題に共通する
テンソル化・次元と有限値の確認・コピーを行います。

## 経験の保存

通常の `choose_action` は観測、状態、行動インデックスと選択時の対数確率を
`pending` に保持します。続く `learn` が報酬、次の観測と状態、ゴール到達の
有無を加え、1遷移として `trajectory` に保存します。`end_episode` は
ゴール到達や時間切れで軌跡を `episodes` に移します。評価時の
`choose_action(..., greedy=True)` は経験を保存しません。
到達ゴールはアダプターの `encode_achieved_goal` で変換します。これを持たない
問題では次状態を使うため、その次元は `goal_dim` と一致する必要があります。
`settings` の `achieved_goal_encoder` でも変換関数を指定できます。
計算デバイスはWeb UIまたは `grag_device`（`cpu`、`cuda`、`cuda:N`、`auto`）で
指定できます。直接利用する場合は `device` でも指定できます。既定値はCUDAです。
対照学習の学習率は `contrastive_learning_rate`（既定0.0003）です。
`grag_contrastive_epochs`（既定1、範囲1〜1000）で、同じHERバッチを使う更新回数を指定します。
HER目標はバッチごとに一度抽選し、各エポックでは更新後のエンコーダで損失を計算し直します。
最初の10,000遷移は収集に使い、その後に分割して連続更新します。
以降は `grag_update_interval` 遷移ごとにPPOと対照学習を更新します。
対照学習では各遷移について、同一エピソードの現在以降の到達状態を
`hindsight_gamma` で重み付けし、目標を1件抽出します。
HERの後はバッチ全体で目標ごとに正例の状態・行動をまとめます。他の目標グループにだけ
属する状態・行動を負例とし、単一エピソードの経験でも比較できる目標があれば更新します。
複数の目標に属する状態・行動は、所属する各目標で正例として扱います。
目標を固定して状態・行動に対してsoftmaxを取り、正例の負の対数確率を平均します。
エンコーダは1エポックにつき1回更新し、最後のエポックの損失を記録します。
負例がなければ経験を蓄積します。
PPOは収集時の目標に加え、対照学習と同じHER目標も使います。元の目標と一致する
HER目標は各遷移で一度だけ数えます。損失を目標ごとに平均し、各目標を同じ重みで
まとめます。GridWorldではActorの観測に含まれる目標も差し替え、確率比の分母には
行動収集時の対数確率を使います。他の環境でも、アダプターの
`relabel_goal(observation, goal)` でActorに渡す観測を差し替えられます。

PPOでは、指定ゴールの表現と状態・行動の表現の負のユークリッド距離を
Q値として使います。全行動の方策期待値をベースラインとし、その差を
advantageとしてGAEなしで計算します。収集時の対数確率を使うクリップ付き
方策損失とエントロピー項でActorのみを更新します。目標はアダプターの
`encode_goal` または設定の `target_goal_encoder` で取り出し、どちらもない
場合は状態をゴールとして使います（`goal_dim` と状態の次元が一致する必要があります）。
クリップ幅は0.2、エントロピー係数は0.01です。

