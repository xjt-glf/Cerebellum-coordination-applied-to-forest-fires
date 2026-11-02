# -*- coding: utf-8 -*-
"""大小脑协同灭火仿真 · 本地可视化服务。

    .venv/bin/python webapp/server.py          # 然后打开 http://127.0.0.1:8000

低层用训练好的 MAPPO 权重（models/actor_selected.pt），
高层默认用启发式规划器，也可以在页面上填 OpenAI 兼容的接口换成大模型。
仿真在后台线程里跑，前端轮询取帧后在 canvas 上回放。

注意：服务默认只监听 127.0.0.1，没有任何鉴权。它会把前端填入的 api_key
转发给你指定的 base_url，所以不要把 --host 改成 0.0.0.0 暴露到公网。
"""
import base64
import json
import math
import os
import sys
import threading
import time
import uuid

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

DEFAULT_ACTOR = os.path.join(_ROOT, "models", "actor_selected.pt")

app = FastAPI(title="大小脑协同灭火仿真")
_SESS = {}
_LOCK = threading.Lock()


# ----------------------------------------------------------------- 网格编码
def fire_code(bs):
    """真值火场 → uint8：0 健康 / 1 燃烧中 / 2 完全燃烧 / 3 烧尽 / 4 已扑灭。

    与 Env/低层环境.py 的 render() 配色语义一致。
    """
    c = np.zeros(bs.shape, np.uint8)
    c[(bs > 0) & (bs < 1)] = 1
    c[bs == 1.0] = 2
    c[np.isclose(bs, 2.0)] = 3
    c[np.isclose(bs, 3.0)] = 4
    return c


def known_code(mi):
    """认知地图（-1 为未知）→ uint8：0 未知 / 1 健康 / 2 有火 / 3 烧尽 / 4 已扑灭。"""
    c = np.zeros(mi.shape, np.uint8)
    c[mi == 0] = 1
    c[(mi > 0) & (mi <= 1.0)] = 2
    c[np.isclose(mi, 2.0)] = 3
    c[np.isclose(mi, 3.0)] = 4
    return c


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def quant(a):
    """浮点场 → uint8（线性拉伸到 0-255），只用于前端画背景贴图。"""
    a = np.asarray(a, np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)



def delta(prev, cur):
    """只传变化的格子，600 步整局的网格数据因此压到几百 KB。"""
    idx = np.flatnonzero(prev.ravel() != cur.ravel())
    if idx.size == 0:
        return None
    return {"i": idx.astype(np.int32).tolist(),
            "v": cur.ravel()[idx].astype(np.uint8).tolist()}


# ----------------------------------------------------------------- 请求模型
class RunCfg(BaseModel):
    # 环境（默认值与 deliverable/simulate.py 一致）
    grid: int = 100
    fires: int = 3
    random_fires: bool = False
    rescuers: int = 10
    uavs: int = 4
    steps: int = 600
    spread_steps: int = 40
    spread_gain: float = 1.0
    seed: int = 20000
    personnel_interactive: bool = False  # 每个高层窗口等待明确人员指令
    # 人员交互演示默认不调用外部API；用户可显式选择llm。
    brain: str = "heuristic"          # heuristic | llm | agent (外部 Agent 联合规划)
    replan_every: int = 20
    explore_until: float = 0.70
    charge_at: float = 300.0
    max_chargers: int = 2
    base_url: str = ""  # 在前端输入兼容 OpenAI 的服务端点
    api_key: str = ""  # 在前端输入，不包含默认凭据
    model: str = "qwen-plus"
    temperature: float = 0.2
    llm_timeout: float = 180.0
    llm_full_state: bool = True       # 包括完整真值格图，用于充分信息测试
    # 运行
    every: int = 2                    # 每几步录一帧
    random_policy: bool = False       # 用随机动作替代策略，作为对照基线
    waypoint_assist: bool = False     # 定点驻留指令改用确定性寻路（见 _waypoint_action）
    goal_update: bool = True          # 完成/失效后更新目标，不接管动作
    stop_when_clear: bool = True      # 活火点归零就提前收队，不跑满步数上限
    stop_when_depleted: bool = False  # 全机耗尽可选择提前结束；默认仍跑满步数


# 环境的 8 个移动方向，顺序必须与 Env/低层环境.py:2222 一致
_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def _waypoint_action(u, tgt, hold=2.0):
    """朝 tgt 走一步的确定性动作；已经在 hold 格以内就悬停。

    只用于"定点驻留"指令（relay + 显式 target + 无 rescuer_id）。实测低层策略
    对任意坐标的驻留指令跟随很不稳定（7 个目标里 3 个能收敛到 4 格内，4 个停在
    20-50 格外），所以给一个可选的确定性兜底：被事件钉死的那一架交给寻路保证
    到位，其余飞机仍然完全由 MAPPO 策略控制。
    """
    dx, dy = float(tgt[0]) - float(u.x), float(tgt[1]) - float(u.y)
    if math.hypot(dx, dy) <= hold:
        return 8
    best, bd = 8, float("inf")
    for k, (ax, ay) in enumerate(_DIRS):
        d = math.hypot(dx - ax, dy - ay)
        if d < bd:
            best, bd = k, d
    return best


def _is_hold_point(o):
    """这条指令是不是"去某个坐标驻留"。"""
    return (o is not None and int(getattr(o, "task", 0)) == 1
            and getattr(o, "target", None) is not None
            and getattr(o, "rescuer_id", None) is None)


def _active_burning(e):
    """真值里**未合围**的燃烧格数。0 就代表"没有活火点了"：剩下的火要么被扑灭、
    要么已烧尽、要么被隔火带围住不会再扩散（Env/低层环境.py:1229 同一个判据，
    评估口径的 active_discovery_rate 也是用它算分母的）。"""
    bs = e.grid["burning_state"]
    f = getattr(e, "_active_true_burning_mask", None)
    if f is not None:
        try:
            return int(f(bs).sum())
        except Exception:
            pass
    return int(((bs > 0) & (bs < 2)).sum())


def _fire_counts(e):
    """结束判据包含所有仍燃烧的格子（含已合围），以及人员图中的残留火情。"""
    bs = e.grid["burning_state"]
    people = e.rescuers_map_info
    return (int(((bs > 0) & (bs < 2)).sum()),
            int(((people > 0) & (people < 2)).sum()))


def _end_reason(step, cfg, options, seen_fire, counts, depleted):
    if step >= cfg.steps:
        return "跑满步数上限"
    if options["stop_when_clear"] and seen_fire and counts == (0, 0):
        return "火情清零（真值及人员认知均无燃烧格）"
    if options["stop_when_depleted"] and depleted >= cfg.uavs:
        return "无人机全部耗尽（已开启提前结束）"
    return ""


METRIC_KEYS = ("explored", "discovery", "rescuer_known", "coverage",
               "burned", "extinguished", "contained", "active_fires",
               "max_aoi", "ret", "depleted", "min_energy")


def _metrics(info, cov_hist, ret, n_act=None):
    m = {
        "explored": round(float(info["explored_rate"]), 4),
        "discovery": round(float(info["fire_discovery_rate"]), 4),
        "rescuer_known": round(float(info["rescuer_known_fire_rate"]), 4),
        "coverage": round(float(np.mean(cov_hist)), 4),
        "burned": round(float(info["burned_area_rate"]), 5),
        "extinguished": int(info["n_extinguished"]),
        "contained": int(info["n_contained"]),
        "active_fires": int(info["n_clusters"]),
        "max_aoi": int(info["max_AoI"]),
        "ret": round(float(ret), 1),
        "depleted": int(info["n_depleted"]),
        "min_energy": round(float(info["min_energy"]), 1),
    }
    # active_fires 是从**全队认知图**里数的（认知会变旧），真值活火格数单独给一项，
    # 否则会出现"提前收队了但 active_fires 还显示 1"这种自相矛盾的报告。
    if n_act is not None:
        m["active_cells"] = int(n_act)
    return m


def _uav_frame(u, i, info, orders, resolved=None):
    """单机的逐帧状态。

    tgt/dt 是"指令跟随"探针。**必须用 resolve_order_target 解析后的目标**，
    而不是 order.target 原值——relay 任务里 rescuer_id 比 target 优先
    （obs_builder.py:266-271），只看原值会显示一个策略根本没在追的坐标。
    """
    d = {"x": float(u.x), "y": float(u.y), "e": round(float(u.e), 1),
         "task": int(info["tasks"][i]), "mode": int(info["modes"][i])}
    o = orders[i] if i < len(orders) else None
    if o is None:
        return d
    # 只有高层**显式**指定过目标才算"下过坐标指令"，否则是策略自己就近挑的兜底目标
    explicit = int(getattr(o, "task", 0)) == 2 or getattr(o, "target", None) is not None or \
        getattr(o, "rescuer_id", None) is not None
    tgt = resolved if resolved is not None else getattr(o, "target", None)
    if explicit and tgt is not None:
        tx, ty = float(tgt[0]), float(tgt[1])
        d["tgt"] = [round(tx, 1), round(ty, 1)]
        d["dt"] = round(math.hypot(tx - float(u.x), ty - float(u.y)), 1)
        raw = getattr(o, "target", None)
        if raw is not None and math.hypot(float(raw[0]) - tx, float(raw[1]) - ty) > 1.0:
            # 下发的坐标被 rescuer_id 顶掉了，前端要如实标出来
            d["tgt_raw"] = [round(float(raw[0]), 1), round(float(raw[1]), 1)]
        rr = getattr(o, "ring_radius", None)
        if rr is not None:
            d["ring"] = float(rr)
    return d


# ----------------------------------------------------------------- 仿真线程
def _worker(sid, cfg: RunCfg):
    import matplotlib
    matplotlib.use("Agg")             # 不开窗口，也不用环境自带的 render()
    import torch
    from lowlevel.env_wrapper import TaskConditionedEnv
    from lowlevel.models import Actor
    from lowlevel.obs_builder import OBS_DIM, shared_cache, resolve_order_target
    from lowlevel.mission_demo import sense_state
    from webapp.brains import make_brain
    from webapp.goal_protocol import GoalProtocol
    from webapp.charge_protocol import ChargeProtocol
    from webapp.world_snapshot import full_snapshot
    from webapp.personnel_control import PersonnelController

    S = _SESS[sid]
    try:
        cwd = os.getcwd()
        os.chdir(_ROOT)               # terrain_height.csv 是相对路径读的
        try:
            env = TaskConditionedEnv(
                num_uavs=cfg.uavs, num_rescuers=cfg.rescuers,
                n_fire_seeds=cfg.fires,
                n_fire_seeds_range=((2, 4) if cfg.random_fires else None),
                grid_size=(cfg.grid, cfg.grid), max_steps=cfg.steps,
                spread_steps=cfg.spread_steps, spread_gain=cfg.spread_gain,
                seed=cfg.seed, replan_every=0,
                terminate_on_all_depleted=False)  # 网页结束条件统一由本服务管理
            env.reset()
        finally:
            os.chdir(cwd)

        actor = None
        if not cfg.random_policy:
            sd = torch.load(DEFAULT_ACTOR, map_location="cpu")
            actor = Actor(OBS_DIM, 9, hidden=int(sd["head.weight"].shape[1]))
            actor.load_state_dict(sd)
            actor.eval()

        brain = make_brain(cfg.model_dump()) if cfg.brain != "agent" else type("JointAgentBrain", (), {"kind": "agent"})()
        S["brain_object"] = brain
        goals = GoalProtocol(env, cfg.seed) if cfg.goal_update else None
        charging = ChargeProtocol(env, goals)
        e = env.env
        personnel = PersonnelController(e)
        e.personnel_controller = personnel
        S["personnel_controller"] = personnel
        H, W = e.grid_size
        f_prev = fire_code(e.grid["burning_state"])
        k_prev = known_code(e.team_map_info)
        r_prev = known_code(e.rescuers_map_info)
        S["init"] = {
            "H": H, "W": W, "steps": cfg.steps, "every": cfg.every, "seed": cfg.seed,
            "n_uavs": cfg.uavs, "n_rescuers": cfg.rescuers,
            "stations": [[int(x), int(y)] for x, y in e.charging_stations],
            "base": [float(e.base_pos[0]), float(e.base_pos[1])],
            "base_r": float(e.base_comm_radius),
            "uav_comm_r": float(e.uav_comm_radius),
            "brain": brain.kind, "replan_every": cfg.replan_every,
            "policy": "随机动作基线" if cfg.random_policy else "MAPPO " + os.path.basename(DEFAULT_ACTOR),
            "goal_update": cfg.goal_update,
            "personnel_interactive": cfg.personnel_interactive,
            "people0": [personnel.person_frame(p) for p in e.rescuers],
            "llm_input_mode": "完整环境（含真值）" if cfg.llm_full_state else "态势简报",
            "fire0": b64(f_prev), "known0": b64(k_prev), "rknown0": b64(r_prev),
            # 背景贴图用的真实地形与可燃物分布（线性量化到 0-255）
            "terrain": b64(quant(e.grid["height"])),
            "fuel": b64(quant(e.grid["flammability"])),
        }
        S["status"] = "running"

        rng = np.random.RandomState(cfg.seed)
        obs, ret, cov = None, 0.0, []
        info = None
        t = 0
        seen_fire = _fire_counts(e)[0] > 0
        steps_run = 0
        end_reason = ""
        for t in range(cfg.steps):
            if S["stop"]:
                break
            if (cfg.personnel_interactive or cfg.brain == "agent") and t % cfg.replan_every == 0:
                with _LOCK:
                    S["personnel_request"] = personnel.snapshot(t)
                    if cfg.brain == "agent":
                        S["personnel_request"]["situation"] = sense_state(env)
                        S["personnel_request"]["current_uav_orders"] = [o.to_dict() for o in env.orders]
                        S["personnel_request"]["unknown_tiles"] = [
                            [x, y, int((e.team_map_info[x:x+20, y:y+20] < 0).sum())]
                            for x in range(0,H,20) for y in range(0,W,20)]
                    S["status"] = "waiting_personnel"
                while not S["stop"]:
                    S["wake"].wait(.2)
                    with _LOCK:
                        ready = bool(S["personnel_submissions"])
                    if ready:
                        break
                if S["stop"]:
                    break
            with _LOCK:
                submissions = S["personnel_submissions"][:]
                S["personnel_submissions"].clear()
                S["wake"].clear()
            for submission in submissions:
                personnel.enqueue(submission["orders"], t, submission["source"], submission["reason"])
            S["status"] = "running"
            if t % cfg.replan_every == 0:
                s = sense_state(env)
                if cfg.brain == "llm" and cfg.llm_full_state:
                    s["full_environment"] = full_snapshot(env, goals, S["log"])
                ev = S.get("event") or None
                t0 = time.time()
                if cfg.brain == "agent":
                    joint = submissions[-1]
                    plan = {"uavs": joint["uavs"]}
                    reason = [joint["reason"]]
                    meta = {"source": "codex_joint", "external_api_calls": 0}
                else:
                    plan, reason, meta = brain.plan(s, event=ev)
                requested_plan = plan
                if goals is not None:
                    obs, plan, goal_notes = goals.apply(
                        plan, source=meta.get("source", brain.kind), event=ev)
                    reason = list(reason) + goal_notes
                else:
                    obs = env.set_plan(plan)
                S["log"].append({"t": t, "plan": plan, "reason": reason,
                                 "personnel_orders": [o for sub in submissions for o in sub["orders"]],
                                 "requested_plan": requested_plan,
                                 "meta": meta, "ms": int((time.time() - t0) * 1000),
                                 "event": ev,
                                 "brief": {k: s[k] for k in
                                           ("explored", "discovery", "coverage",
                                            "rescuer_known", "burned", "max_aoi")},
                                 "n_clusters": len(s["clusters"])})
            charge_obs, charge_notes = charging.sync()
            if charge_obs is not None:
                obs = charge_obs
            if charge_notes:
                applied = {"uavs": [order.to_dict() for order in env.orders]}
                if S["log"] and S["log"][-1]["t"] == t:
                    S["log"][-1]["plan"] = applied
                    S["log"][-1]["reason"].extend(charge_notes)
                else:
                    s = sense_state(env)
                    S["log"].append({"t": t, "plan": applied, "reason": charge_notes,
                                     "meta": {"source": "charging_guard"}, "ms": 0,
                                     "event": S.get("event"),
                                     "brief": {k: s[k] for k in
                                               ("explored", "discovery", "coverage",
                                                "rescuer_known", "burned", "max_aoi")},
                                     "n_clusters": len(s["clusters"])})
            if actor is None:
                a = rng.randint(0, 9, size=cfg.uavs)
            else:
                with torch.no_grad():
                    a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                                  deterministic=True)[0].numpy()
            if cfg.waypoint_assist:
                # 只接管"定点驻留"那几架，其余仍由策略输出
                a = np.asarray(a).copy()
                for i, u in enumerate(e.uavs):
                    o = env.orders[i] if i < len(env.orders) else None
                    if _is_hold_point(o) and float(u.e) > 0.0:
                        a[i] = _waypoint_action(u, o.target)
            obs, r, done, info = env.step(a)
            if goals is not None:
                goals.update()
            ret += float(np.sum(r))
            cov.append(info["coverage_rate"])

            n_act = _active_burning(e)
            counts = _fire_counts(e)
            seen_fire = seen_fire or counts[0] > 0
            steps_run = t + 1
            end_reason = _end_reason(steps_run, cfg, S["stop_options"],
                                     seen_fire, counts, int(info["n_depleted"]))

            if t % cfg.every == 0 or end_reason:
                f_cur = fire_code(e.grid["burning_state"])
                k_cur = known_code(e.team_map_info)
                r_cur = known_code(e.rescuers_map_info)
                orders = getattr(env, "orders", None) or []
                cch = shared_cache(e)
                res = []
                for i in range(len(e.uavs)):
                    try:
                        res.append(resolve_order_target(e, i, orders[i], cch)[0]
                                   if i < len(orders) else None)
                    except Exception:
                        res.append(None)
                fr = {
                    "t": t + 1,
                    "uavs": [_uav_frame(u, i, info, orders, res[i])
                             for i, u in enumerate(e.uavs)],
                    "res": [personnel.person_frame(p) for p in e.rescuers],
                    "m": _metrics(info, cov, ret, n_act),
                }
                fr["m"].update(burning_cells=counts[0], people_burning_cells=counts[1])
                for i, u in enumerate(e.uavs):
                    fr["uavs"][i]["stranded"] = bool(u.e <= 0 and not e.at_station(u.x, u.y))
                    fr["uavs"][i]["recharged"] = bool(u.recharged)
                for key, pv, cv in (("fire", f_prev, f_cur), ("known", k_prev, k_cur),
                                    ("rknown", r_prev, r_cur)):
                    d = delta(pv, cv)
                    if d is not None:
                        fr[key] = d
                f_prev, k_prev, r_prev = f_cur, k_cur, r_cur
                with _LOCK:
                    S["frames"].append(fr)
                    S["personnel_request"] = personnel.snapshot(t + 1)
            if end_reason:
                break

        S["result"] = (_metrics(info, cov, ret, _active_burning(e))
                       if info is not None else {})
        S["result"].update(burning_cells=_fire_counts(e)[0],
                            people_burning_cells=_fire_counts(e)[1])
        S["result"]["steps"] = steps_run
        S["result"]["personnel_commands"] = personnel.events
        S["result"]["stop_options"] = dict(S["stop_options"])
        S["result"]["end_reason"] = ("用户停止" if S["stop"] else
                                     end_reason or "跑满步数上限")
        if getattr(brain, "kind", "") == "llm":
            S["result"]["llm_calls"] = brain.n_call
            S["result"]["llm_failures"] = brain.n_fail
        S["status"] = "stopped" if S["stop"] else "done"
        if cfg.brain == "agent":
            replay_dir = os.path.join(_ROOT, "research_outputs", "webapp_replays")
            os.makedirs(replay_dir, exist_ok=True)
            replay = {k: S.get(k) for k in ("status", "init", "frames", "log", "result")}
            replay["personnel"] = S.get("personnel_request")
            with open(os.path.join(replay_dir, sid + ".json"), "w") as handle:
                json.dump(replay, handle, ensure_ascii=False)
    except Exception as ex:          # 让前端看到真实报错，而不是干等
        import traceback
        S["status"] = "error"
        S["error"] = f"{type(ex).__name__}: {ex}"
        S["trace"] = traceback.format_exc()[-3000:]


# -------------------------------------------------------------------- 路由
@app.post("/api/start")
def api_start(cfg: RunCfg):
    if not (1 <= cfg.uavs <= 8):
        raise HTTPException(400, "无人机数量必须在 1-8 之间（观测里 agent-id one-hot 上限 8）")
    if not cfg.random_policy and not os.path.exists(DEFAULT_ACTOR):
        raise HTTPException(500, f"找不到低层权重 {DEFAULT_ACTOR}")
    if cfg.brain == "llm" and not (cfg.base_url and cfg.model):
        raise HTTPException(400, "选了大模型大脑，必须填 base_url 和 model")
    if cfg.replan_every < 1:
        raise HTTPException(400, "重规划周期必须 >= 1")
    if cfg.steps < 1 or cfg.every < 1:
        raise HTTPException(400, "回合步数和录帧间隔必须 >= 1")
    sid = uuid.uuid4().hex[:12]
    _SESS[sid] = {"status": "starting", "frames": [], "log": [], "init": None,
                  "result": None, "stop": False, "error": None, "trace": None,
                  "stop_options": {"stop_when_clear": cfg.stop_when_clear,
                                   "stop_when_depleted": cfg.stop_when_depleted},
                  "event": None}
    _SESS[sid].update(personnel_submissions=[], personnel_request=None,
                      personnel_controller=None, brain_kind=cfg.brain, n_uavs=cfg.uavs, wake=threading.Event())
    th = threading.Thread(target=_worker, args=(sid, cfg), daemon=True)
    _SESS[sid]["thread"] = th
    th.start()
    # 清掉旧会话，避免内存堆积
    for k in [k for k in list(_SESS) if k != sid
              and _SESS[k]["status"] in ("done", "stopped", "error")][:-3]:
        _SESS.pop(k, None)
    return {"sid": sid}


@app.get("/api/frames/{sid}")
def api_frames(sid: str, since: int = 0, log_since: int = 0):
    S = _SESS.get(sid)
    if S is None:
        if len(sid) == 12 and all(c in "0123456789abcdef" for c in sid):
            replay_path = os.path.join(_ROOT, "research_outputs", "webapp_replays", sid + ".json")
            if os.path.isfile(replay_path):
                with open(replay_path) as handle:
                    replay = json.load(handle)
                all_frames, all_log = replay["frames"], replay["log"]
                replay.update(frames=all_frames[max(0,since):max(0,since)+400],
                              log=all_log[max(0,log_since):], total=len(all_frames), n_log=len(all_log))
                if since != 0:
                    replay.pop("init", None)
                return JSONResponse(replay)
        raise HTTPException(404, "会话不存在（服务重启过？重新点开始）")
    with _LOCK:
        frames = S["frames"][since:since + 400]
    out = {"status": S["status"], "total": len(S["frames"]),
           "frames": frames, "log": S["log"][log_since:],
           "n_log": len(S["log"]), "result": S["result"],
           "event": S.get("event")}
    out["personnel"] = S.get("personnel_request")
    if since == 0 and S["init"] is not None:
        out["init"] = S["init"]
    if S["error"]:
        out["error"], out["trace"] = S["error"], S["trace"]
    return JSONResponse(out)


class EventIn(BaseModel):
    text: str = ""


class PersonnelIn(BaseModel):
    uavs: list[dict] | None = None
    orders: list[dict] = Field(default_factory=list)
    source: str = "operator"
    reason: str = ""
    expected_tick: int | None = None


@app.get("/api/personnel/{sid}")
def api_personnel_state(sid: str):
    session = _SESS.get(sid)
    if session is None:
        raise HTTPException(404, "会话不存在")
    return {"status": session["status"], "snapshot": session.get("personnel_request")}


@app.post("/api/personnel/{sid}")
def api_personnel_command(sid: str, command: PersonnelIn):
    session = _SESS.get(sid)
    if session is None:
        raise HTTPException(404, "会话不存在")
    with _LOCK:
        if session["status"] not in {"running", "waiting_personnel"}:
            raise HTTPException(409, "会话未运行；请先开始新一局")
        controller = session.get("personnel_controller")
        if controller is None:
            raise HTTPException(409, "环境尚未准备好")
        tick = (session.get("personnel_request") or {}).get("t", 0)
        if command.expected_tick is not None and command.expected_tick != tick:
            raise HTTPException(409, f"指令时刻过期：当前 t={tick}")
        if session["status"] == "waiting_personnel" and session["personnel_submissions"]:
            raise HTTPException(409, "当前窗口已有待处理指令")
        try:
            orders = controller.normalize(command.orders)
            if session.get("brain_kind") == "agent":
                if not command.uavs or len(command.uavs) != session["n_uavs"]:
                    raise ValueError("Agent 联合规划必须包含全部无人机指令")
                for order in command.uavs:
                    if order.get("task") not in {"explore", "relay", "charge", "patrol"}:
                        raise ValueError("未知无人机任务")
                    target = order.get("target")
                    if target is not None and (not isinstance(target, list) or len(target) != 2
                            or not all(isinstance(v, (int,float)) and math.isfinite(v)
                                       and 0 <= v < controller.env.grid_size[i] for i,v in enumerate(target))):
                        raise ValueError("无人机目标坐标无效")
                    if order["task"] == "explore" and target is None:
                        raise ValueError("探索任务必须有航点")
        except (ValueError, TypeError, OverflowError) as exc:
            raise HTTPException(422, str(exc)) from exc
        session["personnel_submissions"].append(dict(orders=orders, uavs=command.uavs, source=command.source[:64],
                                                    reason=command.reason[:2000]))
        session["wake"].set()
    return {"ok": True, "queued": orders, "t": tick,
            "note": "进入控制队列；沿用现有通信覆盖判定，连通后激活"}


class StopOptions(BaseModel):
    stop_when_clear: bool
    stop_when_depleted: bool = False


@app.post("/api/stop_options/{sid}")
def api_stop_options(sid: str, options: StopOptions):
    S = _SESS.get(sid)
    if S is None:
        raise HTTPException(404, "会话不存在")
    S["stop_options"] = options.model_dump()
    return {"ok": True, **S["stop_options"]}


@app.post("/api/event/{sid}")
def api_event(sid: str, ev: EventIn):
    """注入/清除一条自然语言突发事件。它只改变**下一次及之后**重规划时喂给
    大模型的 prompt，不修改环境真值——不会凭空多出火点或救援人员。"""
    S = _SESS.get(sid)
    if S is None:
        raise HTTPException(404, "会话不存在")
    txt = (ev.text or "").strip()
    S["event"] = txt or None
    return {"ok": True, "event": S["event"]}


@app.post("/api/stop/{sid}")
def api_stop(sid: str):
    S = _SESS.get(sid)
    if S is None:
        raise HTTPException(404, "会话不存在")
    S["stop"] = True
    S["wake"].set()
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(os.path.join(_HERE, "static", "index.html"))


@app.get("/api/model")
def api_model():
    return {"actor": os.path.basename(DEFAULT_ACTOR),
            "path": os.path.relpath(DEFAULT_ACTOR, _ROOT)}


@app.get("/api/prompt/{sid}")
def api_prompt(sid: str):
    session = _SESS.get(sid)
    if session is None:
        raise HTTPException(404, "会话不存在")
    prompt = getattr(session.get("brain_object"), "last_prompt", "")
    if not prompt:
        raise HTTPException(404, "本局尚未生成大模型输入；请使用大模型运行")
    return PlainTextResponse(prompt)


if __name__ == "__main__":
    import argparse
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--actor", default=DEFAULT_ACTOR, help="Actor 权重路径")
    a = p.parse_args()
    DEFAULT_ACTOR = os.path.abspath(a.actor)
    if not os.path.isfile(DEFAULT_ACTOR):
        p.error("Actor 权重不存在")
    print(f"打开 http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")



