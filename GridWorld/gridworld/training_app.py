"""学習をその場で観察し、任意の回数だけ高速学習するアプリ（5003）。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import re
import math
from threading import Event, Lock
from time import monotonic
from uuid import uuid4
from urllib.parse import unquote

from agents.q_learning import QLearningAgent
from agents.cppo import CPPOAgent
from agents.sgcrl import GridWorldSGCRL
from gridworld.storage import (RESULT, WORLD, list_maps, load_settings,
                     map_path, read_json, validate_map, validate_settings, write_json)
from gridworld.training_runtime import TrainingSession
from gridworld.training_runtime.ppo_adapter import GRAGSession, PPOAgentSession
from gridworld.web_common import Handler, serve


LEARNING_AGENTS = {"q_learning": QLearningAgent, "ppo": PPOAgentSession,
                   "grag": GRAGSession, "sgcrl": GridWorldSGCRL, "cppo": CPPOAgent}
PPO_MODES = {"ppo", "grag", "sgcrl", "cppo"}
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
                job["preview_revision"] = job.get("preview_revision", 0) + 1
                job["completed"] = max(0, session.completed_episodes - job["episodes_before"])
                if job.get("benchmark"):
                    job["elapsed_seconds"] = session.training_seconds
            last_publish = now

    def save():
        publish(force=True)
        with job_lock:
            job["saving"] = True
            job["phase"] = "学習を停止して保存中です"
        result = session_result(session, job["session_id"], mode, checkpoint=True)
        write_json(saved_session_path(job["session_id"]), result)
        # 古い観察ジョブを閉じて新規学習を始めた場合、最新結果を取り違えない。
        with job_lock:
            is_current = current_job is job
        if is_current:
            write_json(RESULT, display_result(result) if mode in PPO_MODES else result)
        with job_lock:
            job["saving"] = False
        return result

    try:
        if mode in PPO_MODES:
            threads = configure_ppo_cpu_threads()
            with job_lock:
                job["cpu_threads"] = threads

        if job.get("benchmark"):
            # 常に新規学習から開始し、100/100成功・上限・停止要求のいずれかで終了する。
            publish(force=True)
            while True:
                with job_lock:
                    stopped = job["_control"] == "paused"
                if stopped:
                    outcome = "cancelled"
                    break
                session.step()
                if session.success_100_milestone is not None:
                    outcome = "reached"
                    break
                if session.training_seconds >= job["max_hours"] * 3600:
                    outcome = "limit"
                    break
                publish()
            session.benchmark.update(status=outcome, training_seconds=session.training_seconds,
                                     completed_episodes=session.completed_episodes,
                                     total_steps=session.total_steps)
            session.evaluate()
            save()
            with job_lock:
                job.update(status="complete", completed=session.completed_episodes,
                           benchmark_outcome=outcome, phase={"reached": "100%到達・計測完了",
                           "limit": "時間上限・未達成", "cancelled": "計測を停止"}[outcome])
                job.pop("preview", None)
            return

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
                    job.update(status="paused", execution="paused", phase="一時停止", control_pending=False)
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
        if job.get("benchmark"):
            session.benchmark["status"] = "error"
        with job_lock:
            job.update(status="error", error=str(error), saving=False)
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
                entry = learning_sessions.get(status.get("session_id"))
            if entry is not None:
                session = entry["session"]
                status["training_clock"] = session.training_clock()
                runtime = getattr(session.agent, "runtime_status", None)
                if callable(runtime):
                    status["runtime"] = runtime()
            return self.json(status)
        return super().get_api(path)

    def post_api(self, path, body):
        global current_job
        request = self.parse_json(body)
        if path in ("/api/train/policy", "/api/train/grag-values"):
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
            if path == "/api/train/grag-values":
                try:
                    data = session.inspect_grag_values()
                except Exception:
                    data = {"values": None, "warning": "価値ヒートマップを取得できませんでした。"}
            else:
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
                if job.get("benchmark") and action != "pause":
                    raise ValueError("計測中は設定や実行方法を変更できません。停止してから新しく開始してください。")
                if action == "pause":
                    job["_control"] = "paused"
                    job["control_pending"] = True
                    job["phase"] = "停止要求を受け付けました。更新の完了を待っています"
                elif action == "observe":
                    job["_control"] = "observe"
                    job["control_pending"] = False
                    job["total"] = 0
                elif action == "skip":
                    job["control_pending"] = False
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
            if execution not in ("batch", "observe", "fast", "paused", "benchmark"):
                raise ValueError("実行方法が正しくありません。")
            benchmark = execution == "benchmark"
            max_hours = request.get("max_hours", 1)
            if benchmark and (isinstance(max_hours, bool) or not isinstance(max_hours, (int, float))
                              or not math.isfinite(max_hours) or not 0 < max_hours <= 100000):
                raise ValueError("計測の最大時間は0より大きく100,000以下の時間で指定してください。")
            if benchmark and (evaluation_only or request.get("session_id") is not None):
                raise ValueError("100%までの計測は既存の学習を引き継がず、新規学習から開始します。")
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
                        if benchmark:
                            session.benchmark = {"status": "running", "window": 100,
                                                 "max_hours": max_hours, "start_episode": 0}
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
                       "benchmark": benchmark,
                       "max_hours": max_hours if benchmark else None,
                       "elapsed_seconds": 0,
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
    from gridworld.training_ui import main
    main()
