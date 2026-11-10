"""学習をその場で観察し、任意の回数だけ高速学習するアプリ（5003）。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import re
from threading import Event, Lock
from time import monotonic
from uuid import uuid4
from urllib.parse import unquote

from agents.q_learning import QLearningAgent
from gridworld.storage import (RESULT, WORLD, list_maps, load_settings,
                     map_path, read_json, validate_map, validate_settings, write_json)
from gridworld.training_runtime import TrainingSession
from gridworld.training_runtime.ppo_adapter import GRAGSession, PPOAgentSession
from gridworld.web_common import Handler, serve


LEARNING_AGENTS = {"q_learning": QLearningAgent, "ppo": PPOAgentSession,
                   "grag": GRAGSession}
PPO_MODES = {"ppo", "grag"}
DEFAULT_SKIP_EPISODES = 100
learning_sessions = {}
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grid-training")
job_lock = Lock()
current_job = None


def canonical_agent_mode(mode):
    """旧MyAgentのAPI・保存データの識別名をGRAGへ読み替える。"""
    return "grag" if mode == "myagent_ppo" else mode


def configure_ppo_cpu_threads():
    """計算するスレッド自身に、論理CPU数−2の上限を適用する。"""
    import torch
    threads = max(1, (os.cpu_count() or 1) - 2)
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() > threads:
        try:
            torch.set_num_interop_threads(threads)
        except RuntimeError:
            # inter-opの設定はプロセス内で一度だけ変更できる。
            pass
    return torch.get_num_threads()


def display_result(result: dict) -> dict:
    if "mode" in result:
        result = {**result, "mode": canonical_agent_mode(result["mode"])}
    history = result["history"]
    stride = max(1, (len(history) + 699) // 700)
    sampled = history[::stride]
    if sampled and sampled[-1] is not history[-1]:
        sampled.append(history[-1])
    private = {"recent_rewards", "recent_steps", "episode_state"}
    return {**{key: value for key, value in result.items() if key not in private},
            "history": sampled, "q_table": None, "agent_checkpoint": None}


def session_result(session, session_id, mode, *, compact=False, checkpoint=False):
    result = session.result(compact=compact, checkpoint=checkpoint)
    result.update({"session_id": session_id, "mode": mode, "can_continue": True,
                   "policy_episode": session.completed_episodes})
    return result


def saved_session_path(session_id):
    if not isinstance(session_id, str) or not re.fullmatch(r"[0-9a-f]{32}", session_id):
        raise ValueError("学習セッションIDが正しくありません。")
    return RESULT.parent / "training_sessions" / f"{session_id}.json"


def find_session(session_id, mode, selected):
    mode = canonical_agent_mode(mode)
    path = saved_session_path(session_id)
    entry = learning_sessions.get(session_id)
    if entry is None:
        if not path.is_file():
            raise ValueError("続行する学習がありません。まず新しい学習を始めてください。")
        saved = read_json(path)
        if canonical_agent_mode(saved.get("mode")) != mode:
            raise ValueError("続行時は同じAIを選択してください。")
        if mode in PPO_MODES:
            configure_ppo_cpu_threads()
        session = TrainingSession.restore(saved, LEARNING_AGENTS[mode])
        entry = {"mode": mode, "session": session}
        learning_sessions[session_id] = entry
    if entry["mode"] != mode:
        raise ValueError("続行時は同じAIを選択してください。")
    if entry["session"].world.map != selected:
        raise ValueError("続行時は同じマップを選択してください。変更した場合は新しい学習を始めてください。")
    return entry["session"]


def public_job(job):
    return {key: value for key, value in job.items() if not key.startswith("_")}


def validate_episodes(value):
    if type(value) is not int or not 1 <= value <= 1_000_000:
        raise ValueError("高速学習する試行回数は1〜1,000,000の整数にしてください。")
    return value


def validate_delay(value):
    if type(value) is not int or not 20 <= value <= 2000:
        raise ValueError("観察速度は1行動あたり20〜2000ミリ秒にしてください。")
    return value


def run_job(job: dict, selected: dict, settings: dict, mode: str) -> None:
    session = (learning_sessions[job["session_id"]]["session"]
               if mode in LEARNING_AGENTS else None)
    last_publish = 0.0

    def publish(force=False):
        nonlocal last_publish
        now = monotonic()
        if session is not None and (force or now - last_publish >= 0.1):
            preview = session_result(session, job["session_id"], mode, compact=True)
            with job_lock:
                job["preview"] = preview
                job["completed"] = max(0, session.completed_episodes - job["episodes_before"])
            last_publish = now

    def save():
        result = session_result(session, job["session_id"], mode, checkpoint=True)
        write_json(saved_session_path(job["session_id"]), result)
        # 古い観察ジョブを閉じて新規学習を始めた場合、最新結果を取り違えない。
        with job_lock:
            is_current = current_job is job
        if is_current:
            write_json(RESULT, result)
        return result

    try:
        if mode in PPO_MODES:
            threads = configure_ppo_cpu_threads()
            with job_lock:
                job["cpu_threads"] = threads

        if not job["_interactive"]:
            def progress(completed):
                publish(force=True)
            if not job.get("evaluation_only"):
                session.run_batch(settings["episodes"], progress=progress)
            session.evaluate()
            save()
            with job_lock:
                job.update(status="complete", completed=job["total"])
                job.pop("preview", None)
            return

        publish(force=True)
        paused_saved = False
        while True:
            with job_lock:
                execution = job["_control"]
                target = job.get("_target", 0)
                save_requested = job.pop("_save_requested", False)
            if execution == "close":
                save()
                with job_lock:
                    job["status"] = "complete"
                    if current_job is not job and current_job.get("session_id") != job["session_id"]:
                        learning_sessions.pop(job["session_id"], None)
                return
            if execution == "paused":
                if not paused_saved or save_requested:
                    save()
                    publish(force=True)
                    paused_saved = True
                with job_lock:
                    job.update(status="paused", execution="paused", phase="一時停止")
                job["_wake"].wait(0.1)
                job["_wake"].clear()
                continue
            paused_saved = False
            if execution == "fast" and session.completed_episodes >= target:
                with job_lock:
                    if job["_control"] == "fast" and job.get("_target") == target:
                        job["last_skip"] = job["total"]
                        job["_control"] = job["_after_fast"]
                save()
                publish(force=True)
                continue
            with job_lock:
                job.update(status="running", execution=execution,
                           phase="観察しながら学習" if execution == "observe" else "高速学習")
                delay = job["step_delay_ms"]
            session.step()
            publish(force=execution == "observe")
            if execution == "observe":
                job["_wake"].wait(delay / 1000)
                job["_wake"].clear()
    except Exception as error:
        with job_lock:
            job.update(status="error", error=str(error))
        # エラー時も公開できる最後の状態は表示する。学習状態は上書きしない。
        publish(force=True)


class TrainingHandler(Handler):
    page = "training.html"

    def get_api(self, path):
        if path in ("/static/training.js", "/static/training.css"):
            name = path.rsplit("/", 1)[-1]
            kind = "application/javascript" if name.endswith(".js") else "text/css"
            return self.file(Path(__file__).resolve().parent / "training_runtime" / name,
                             kind + "; charset=utf-8")
        if path == "/api/maps":
            return self.json(list_maps())
        if path.startswith("/api/maps/"):
            name = unquote(path[len("/api/maps/"):])
            return self.json(validate_map(read_json(map_path(name))))
        if path == "/api/agents":
            return self.json([{"key": key, "name": getattr(agent, "display_name", agent.__name__)}
                              for key, agent in LEARNING_AGENTS.items()])
        if path == "/api/train/config":
            settings = load_settings()
            return self.json({**settings, "episodes": DEFAULT_SKIP_EPISODES})
        if path == "/api/result":
            return self.json(display_result(read_json(RESULT)))
        if path == "/api/train/status":
            with job_lock:
                status = public_job(current_job) if current_job is not None else {"status": "idle"}
            return self.json(status)
        return super().get_api(path)

    def post_api(self, path, body):
        global current_job
        request = self.parse_json(body)
        if path == "/api/train/policy":
            sid = request.get("session_id")
            mode = canonical_agent_mode(request.get("mode"))
            if not isinstance(mode, str) or mode not in LEARNING_AGENTS:
                raise ValueError("行動確率を確認する学習器を選択してください。")
            saved_path = saved_session_path(sid)
            with job_lock:
                entry = learning_sessions.get(sid)
            if entry is None:
                if not saved_path.is_file():
                    raise ValueError("確認する学習がありません。新しい学習を準備してください。")
                saved = read_json(saved_path)
                if canonical_agent_mode(saved.get("mode")) != mode:
                    raise ValueError("行動確率を確認するAIが学習時と一致しません。")
                if mode in PPO_MODES:
                    configure_ppo_cpu_threads()
                # 読み取りだけでは過去のモデルを常駐キャッシュへ追加しない。
                session = TrainingSession.restore(saved, LEARNING_AGENTS[mode])
            else:
                if entry["mode"] != mode:
                    raise ValueError("行動確率を確認するAIが学習時と一致しません。")
                session = entry["session"]
            if request.get("map_name") != session.world.map["name"]:
                raise ValueError("行動確率を確認するマップが学習時と一致しません。")
            if mode in PPO_MODES:
                configure_ppo_cpu_threads()
            data = session.inspect_policy(request.get("position"))
            return self.json({**data, "session_id": sid, "mode": mode,
                              "map_name": session.world.map["name"]})
        if path == "/api/train/control":
            action = request.get("action")
            if action not in ("pause", "observe", "skip", "speed", "limit"):
                raise ValueError("学習操作が正しくありません。")
            episodes = validate_episodes(request.get("episodes")) if action == "skip" else None
            delay = validate_delay(request.get("step_delay_ms")) if action == "speed" else None
            with job_lock:
                job = current_job
                if (job is None or not job.get("_interactive")
                        or job["status"] not in ("running", "paused")):
                    raise ValueError("操作する学習がありません。観察または高速学習を開始してください。")
                if request.get("session_id") != job["session_id"]:
                    raise ValueError("操作する学習が画面と一致しません。画面を更新してください。")
                session = learning_sessions[job["session_id"]]["session"]
                if action == "pause":
                    job["_control"] = "paused"
                elif action == "observe":
                    job["_control"] = "observe"
                    job["total"] = 0
                elif action == "skip":
                    job["_after_fast"] = ("observe" if job["_control"] == "observe" else "paused")
                    job["episodes_before"] = session.completed_episodes
                    job["total"] = episodes
                    job["completed"] = 0
                    job["_target"] = session.completed_episodes + episodes
                    session.last_batch_episodes = episodes
                    job["_control"] = "fast"
                elif action == "limit":
                    session.set_max_steps(request.get("max_steps"))
                    job["_save_requested"] = True
                else:
                    job["step_delay_ms"] = delay
                job["_wake"].set()
                response = public_job(job)
            return self.json(response)

        if path in ("/api/train", "/api/train/evaluate"):
            name = request.get("map_name")
            mode = canonical_agent_mode(request.get("mode", "q_learning"))
            evaluation_only = path == "/api/train/evaluate"
            execution = request.get("execution", "batch")
            if execution not in ("batch", "observe", "fast", "paused"):
                raise ValueError("実行方法が正しくありません。")
            if not isinstance(mode, str) or mode not in LEARNING_AGENTS:
                raise ValueError("AIの種類を選択してください。")
            if not isinstance(name, str) or not map_path(name).is_file():
                raise ValueError("保存済みマップを選択してください。")
            selected = validate_map(read_json(map_path(name)))
            settings = load_settings()
            episodes = (1 if evaluation_only else validate_episodes(
                request.get("episodes", DEFAULT_SKIP_EPISODES)))
            delay = validate_delay(request.get("step_delay_ms", 160))
            extra = request.get("training_settings", {})
            if not isinstance(extra, dict):
                raise ValueError("学習設定はJSONオブジェクトにしてください。")
            if "episodes" in extra:
                raise ValueError("試行回数は高速学習の操作で指定してください。")
            settings.update(extra)
            settings = {**settings, **validate_settings({**settings, "episodes": episodes})}
            sid = request.get("session_id")
            with job_lock:
                previous = current_job
                if previous is not None and previous["status"] == "running":
                    return self.json({"error": "学習中です。一時停止してから新しい学習を始めてください。"}, 409)
                before = 0
                if mode in LEARNING_AGENTS:
                    if evaluation_only and sid is None:
                        raise ValueError("確認する学習済み方策がありません。")
                    if sid is not None:
                        session = find_session(sid, mode, selected)
                        settings = {**session.settings, "episodes": episodes}
                    else:
                        sid = uuid4().hex
                        session = TrainingSession(selected, settings, LEARNING_AGENTS[mode])
                        learning_sessions[sid] = {"mode": mode, "session": session}
                        write_json(RESULT.parent / "training_settings.json", settings)
                    before = session.completed_episodes
                    session.last_batch_episodes = episodes if execution in ("fast", "batch") else 0
                if previous is not None and previous.get("_interactive") and previous["status"] == "paused":
                    previous["_control"] = "close"
                    previous["_wake"].set()
                write_json(WORLD, selected)
                job = {"job_id": uuid4().hex, "status": "running", "mode": mode,
                       "phase": "開始中", "completed": 0,
                       "total": episodes,
                       "workers": 1, "episodes_before": before, "evaluation_only": evaluation_only,
                       "execution": execution, "step_delay_ms": delay,
                       "_interactive": execution != "batch" and mode in LEARNING_AGENTS,
                       "_wake": Event(), "_control": execution if execution != "batch" else "fast",
                       "_target": before + episodes, "_after_fast": "paused"}
                if sid is not None:
                    job["session_id"] = sid
                current_job = job
                # 以前の学習器は保存ファイルから復元できる。常駐するモデルを増やさない。
                keep = {sid, previous.get("session_id") if previous else None}
                for cached_id in list(learning_sessions):
                    if cached_id not in keep:
                        del learning_sessions[cached_id]
                executor.submit(run_job, job, selected, settings, mode)
                response = public_job(job)
            return self.json(response, 202)
        return super().post_api(path, body)


if __name__ == "__main__":
    serve(TrainingHandler, 5003)
