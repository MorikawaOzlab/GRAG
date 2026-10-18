"""環境の固定ルールを確認するアプリ。学習条件はtraining_appで指定する。"""

from gridworld.web_common import Handler, serve


class SettingsHandler(Handler):
    page = "settings.html"

    def get_api(self, path):
        if path == "/api/settings":
            return self.json({
                "actions": ["up", "down", "left", "right"],
                "goal_reward": 1, "other_reward": 0,
                "collision": "位置は変わらず、1行動として数える",
                "coordinates": "0始まりの[行, 列]"})
        return super().get_api(path)

    def post_api(self, path, body):
        if path == "/api/settings":
            raise ValueError("試行回数・最大行動回数・学習パラメータはAI学習アプリで指定してください。")
        return super().post_api(path, body)


if __name__ == "__main__":
    serve(SettingsHandler, 5002)
