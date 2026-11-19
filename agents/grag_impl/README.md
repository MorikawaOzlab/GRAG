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
更新に使用した `episodes` は消去されます。

`grag_update_interval`（既定65,536）遷移以上かつ2エピソード以上を収集し、
エピソード終了時に更新します。更新前にQ値からadvantageを固定し、
バッチ256、既定1エポックで対照損失とPPO損失を同時に更新します。
対照学習は同じエピソードの将来の到達ゴールを正例、他のエピソードの
到達ゴールを負例とするInfoNCEを使います。ミニバッチが1エピソードだけの場合は
別エピソードの負例を追加します。
到達ゴールはアダプターの `encode_achieved_goal` で変換します。これを持たない
問題では次状態を使うため、その次元は `goal_dim` と一致する必要があります。
`settings` の `achieved_goal_encoder` でも変換関数を指定できます。
共通optimizerの初期学習率は `contrastive_learning_rate`（既定0.00025）で、
CPPOと同じ余弦スケジュールで減衰します。更新間隔はWeb UIから変更できます。
将来の到達ゴールは同一エピソードの収集済み遷移から選び、先の時刻ほど
`hindsight_gamma`（既定値 0.99）の累乗で選択確率を下げます。更新時にまだ
収集していない将来の遷移は候補に含まれません。PPOは元の目標を使います。

PPOでは、指定ゴールの表現と状態・行動の表現の負のユークリッド距離を
Q値として使います。全行動の方策期待値をベースラインとし、その差を
advantageとしてGAEなしで計算します。収集時の対数確率を使うクリップ付き
方策損失とエントロピー項でActorのみを更新します。目標はアダプターの
`encode_goal` または設定の `target_goal_encoder` で取り出し、どちらもない
場合は状態をゴールとして使います（`goal_dim` と状態の次元が一致する必要があります）。
クリップ幅は0.2、エントロピー係数は0.01です。

