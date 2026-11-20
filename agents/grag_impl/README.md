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
各遷移の直後に、その行動で到達した状態を正例ゴールとして対照学習します。
同一エピソードの現在までの全遷移を使い、各(s, a)とその直後の結果を正例、
他の遷移の行動・結果との組を負例とします。最初の1遷移は負例がないため更新を待ちます。
`grag_contrastive_epochs`（既定1）回、同じ正例と負例で更新します。
到達ゴールはアダプターの `encode_achieved_goal` で変換します。これを持たない
問題では次状態を使うため、その次元は `goal_dim` と一致する必要があります。
`settings` の `achieved_goal_encoder` でも変換関数を指定できます。
計算デバイスはWeb UIまたは `grag_device`（`cpu`、`cuda`、`cuda:N`、`auto`）で
指定できます。直接利用する場合は `device` でも指定できます。既定値はCUDAです。
対照学習の学習率は `contrastive_learning_rate`（既定0.0003）です。
`grag_update_interval` 遷移ごとにPPOと従来のバッチ対照学習も更新します。
バッチ対照学習は同一エピソードの将来状態を `hindsight_gamma` で重み付けして
抽出し、別エピソードを負例に使います。別エピソードがまだない場合は蓄積します。
両方の対照学習に `grag_contrastive_epochs` が適用されます。
PPOは元の目標を使います。

PPOでは、指定ゴールの表現と状態・行動の表現の内積を
Q値として使います。全行動の方策期待値をベースラインとし、その差を
advantageとしてGAEなしで計算します。収集時の対数確率を使うクリップ付き
方策損失とエントロピー項でActorのみを更新します。目標はアダプターの
`encode_goal` または設定の `target_goal_encoder` で取り出し、どちらもない
場合は状態をゴールとして使います（`goal_dim` と状態の次元が一致する必要があります）。
クリップ幅は0.2、エントロピー係数は0.01です。

