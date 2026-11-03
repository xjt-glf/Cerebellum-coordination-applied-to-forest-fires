# -*- coding: utf-8 -*-
"""配对执行（paired rollout）基础设施：第一方向"指令级建议"的验收基础。

设计要点（对应《创新点探索指南》第 6 节）：

1. **快照**：deepcopy 整个 TaskConditionedEnv + 保存 Python/NumPy/Torch 三套随机状态。
   已核对：默认路径下 Env/低层环境.py 的逐步演化完全确定（火蔓延向量化无随机、
   人员 BFS 移动无随机、random.shuffle 只在 legacy 分支），配合确定性 actor，
   两个分支的唯一差异就是下发的计划本身。
2. **协议**：同一快照出发，分支 B 只在**本段**（第一个 replan 边界）替换计划，
   之后与分支 A 完全相同的协议（同一确定性 brain 每 replan_every 步重规划）。
   这样测的是"一次局部修改在同一状态、同一后续控制协议下的真实任务收益差"，
   不是自举 TD 标签（区别于 read_critic.py 的 _rollout_branch）。
3. **指标**：直接从环境状态计算任务指标向量，不用低层 shaped return
   （set_plan 会 clear_shape，shaping 电位清零会造成虚假回报差）。
   比率类指标（discovery/rknown）分母随活火变化，同时记录原始计数。

用法见 collect_pairs.py。
"""
import copy
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lowlevel.env_wrapper import TaskConditionedEnv
from lowlevel.obs_builder import resolve_order_target, shared_cache, assigned_station
from lowlevel.task_spec import UavOrder, TeamPlan

# ------------------------------------------------------------ 显式基准配置
# 指南 §2 建议初始基准：100x100、4 机、3 火点、10 人、600 步、规划周期 20。
# brain 参数与 deliverable/simulate.py 的默认一致（charge_at=260, explore_until=0.70,
# max_chargers=2）。所有实验必须显式携带这份配置，不依赖散落的默认值。
BENCH = dict(
    grid=100, uavs=4, fires=3, rescuers=10, steps=600,
    spread_steps=40, spread_gain=1.0, replan_every=20,
    explore_until=0.70, charge_at=260.0, max_chargers=2,
    actor=os.path.join(_ROOT, "deliverable", "model", "actor_best.pt"),
)


def make_env(seed, cfg=None):
    c = dict(BENCH, **(cfg or {}))
    return TaskConditionedEnv(
        num_uavs=c["uavs"], num_rescuers=c["rescuers"],
        n_fire_seeds=c["fires"], n_fire_seeds_range=None,
        grid_size=(c["grid"], c["grid"]), max_steps=c["steps"],
        spread_steps=c["spread_steps"], spread_gain=c["spread_gain"],
        seed=seed, task_sampler="planner", replan_every=0,
    )


def load_actor(path=None):
    import torch
    from lowlevel.models import Actor
    from lowlevel.obs_builder import OBS_DIM
    path = path or BENCH["actor"]
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "actor" in sd:
        sd = sd["actor"]
    actor = Actor(OBS_DIM, 9, hidden=int(sd["head.weight"].shape[1]))
    actor.load_state_dict(sd)
    actor.eval()
    return actor


# ------------------------------------------------------------ 指标（全部从状态直算）
def task_metrics(env):
    """任务指标向量。比率类指标分母会随活火变化，原始计数一并给出。"""
    e = env.env
    bs = e.grid["burning_state"]
    n_true = float(((bs > 0) & (bs < 2)).sum())
    return dict(
        t=env.t,
        explored=float(getattr(e, "explored_rate", 0.0)),
        burned_cells=int((bs > 0).sum()),
        burning_cells=int(((bs > 0) & (bs < 2)).sum()),
        extinguished_cells=int((bs == 3).sum()),
        contained=int(getattr(e, "n_contained_clusters", 0)),
        active_clusters=int(getattr(e, "n_active_clusters", 0)),
        discovery=float(getattr(e, "fire_discovery_rate", 0.0)),
        rknown=float(getattr(e, "rescuer_known_fire_rate", 0.0)),
        active_discovery=float(getattr(e, "active_discovery_rate", 0.0)),
        active_rknown=float(getattr(e, "active_rescuer_known_rate", 0.0)),
        coverage=float(getattr(e, "coverage_rate", 0.0)),
        mean_aoi=float(np.mean([r.AoI for r in e.rescuers])) if e.rescuers else 0.0,
        max_aoi=int(getattr(e, "cur_max_AoI", getattr(e, "max_AoI", 0)) or 0),
        depleted=int(sum(1 for u in e.uavs if float(u.e) <= 0.0)),
        min_energy=float(min(u.e for u in e.uavs)),
        n_true_burning=n_true,
    )


METRIC_KEYS = ("explored", "burned_cells", "extinguished_cells", "contained",
               "discovery", "rknown", "depleted", "min_energy")


# ------------------------------------------------------------ 快照
class Snapshot:
    """完整可恢复的环境快照：deepcopy + 三套 RNG 状态。"""

    def __init__(self, env):
        self.env = copy.deepcopy(env)
        self.np_state = np.random.get_state()
        self.py_state = random.getstate()

    def restore(self):
        """返回一个从该快照出发的新环境（再次 deepcopy，可多次 restore）。"""
        env = copy.deepcopy(self.env)
        np.random.set_state(self.np_state)
        random.setstate(self.py_state)
        return env


# ------------------------------------------------------------ 决策上下文（证据特征）
def decision_context(env):
    """重规划边界上的可观测证据（与 sense_state 同源，全是策略可知信息）。

    额外补上：每机的已解析目标与距离、当前指令、返航余量——这是后续
    "依据"特征的输入空间。
    """
    from lowlevel.mission_demo import sense_state
    s = sense_state(env)
    e = env.env
    c = shared_cache(e)
    per_uav = []
    for i, o in enumerate(env.orders):
        tgt, is_fire, ring = resolve_order_target(e, i, o, c)
        u = e.uavs[i]
        d_tgt = float(np.hypot(tgt[0] - u.x, tgt[1] - u.y)) if tgt else None
        per_uav.append(dict(
            i=i, task=int(o.task), mode=int(o.mode),
            target=(None if o.target is None else [round(float(o.target[0]), 1),
                                                   round(float(o.target[1]), 1)]),
            rescuer_id=o.rescuer_id, ring_radius=o.ring_radius,
            handover=bool(o.proactive_handover),
            resolved_target=(None if tgt is None else [round(float(tgt[0]), 1),
                                                       round(float(tgt[1]), 1)]),
            target_is_fire=bool(is_fire), dist_target=d_tgt,
            pos=[round(float(u.x), 1), round(float(u.y), 1)],
            e=round(float(u.e), 1), margin=round(float(u.e) - 2.5 * assigned_station(e, i)[1]),
        ))
    return dict(sense=s, orders=per_uav,
                uncovered_ids=[int(r.rescuer_id) for r in e.rescuers if not r.is_covered])


# ------------------------------------------------------------ 协议 rollout
def run_protocol(env, actor, brain, t_end, override_at=None, override_orders=None,
                 replan_every=20):
    """从当前状态运行到 t_end。

    - 每 replan_every 步（绝对时间网格）用 brain 重规划；t == override_at 的那次
      边界改用 override_orders（局部修改只作用于这一段，之后恢复同一协议）。
    - actor 确定性执行。返回 (结束指标 dict, 窗口内逐步 coverage, 计划应用日志)。
    """
    import torch
    max_steps = env.max_steps
    cov_hist, aoi_hist = [], []
    applied = []
    obs = None
    while env.t < t_end and env.t < max_steps and not getattr(env, "_proto_done", False):
        if env.t % replan_every == 0:
            if override_at is not None and env.t == override_at:
                obs = env.set_plan(list(override_orders))
                applied.append(dict(t=env.t, source="override",
                                    orders=orders_to_json(override_orders)))
            else:
                s = _sense(env)
                plan, reason, meta = brain.plan(s)
                obs = env.set_plan(plan)
                applied.append(dict(t=env.t, source=brain.kind, plan=plan, meta=meta))
        with torch.no_grad():
            a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                          deterministic=True)[0].numpy()
        obs, _r, done, info = env.step(a)
        cov_hist.append(float(info.get("coverage_rate", 0.0)))
        aoi_hist.append(task_metrics(env)["mean_aoi"])
        if done:
            env._proto_done = True          # 全机耗尽等：协议终止，后续调用也不再推进
            break
    m = task_metrics(env)
    m["cov_mean"] = float(np.mean(cov_hist)) if cov_hist else 0.0
    m["aoi_mean"] = float(np.mean(aoi_hist)) if aoi_hist else 0.0
    m["steps_run"] = len(cov_hist)
    return m, cov_hist, applied


def _sense(env):
    from lowlevel.mission_demo import sense_state
    return sense_state(env)


def run_branch(snapshot, actor, brain, t_end, override_at=None, override_orders=None,
               replan_every=20):
    """从快照出发跑一个分支。override_orders 只在 override_at 边界生效。"""
    env = snapshot.restore()
    return run_protocol(env, actor, brain, t_end, override_at=override_at,
                        override_orders=override_orders, replan_every=replan_every)


# ------------------------------------------------------------ H brain（与 webapp 同一实现）
def make_h_brain(cfg=None):
    c = dict(BENCH, **(cfg or {}))
    from webapp.brains import HeuristicBrain
    return HeuristicBrain(explore_until=c["explore_until"], charge_at=c["charge_at"],
                          max_chargers=c["max_chargers"])


# ------------------------------------------------------------ 计划编辑工具
def orders_to_json(orders):
    return [o.to_dict() for o in orders]


def parse_plan(plan_dict, num_uavs):
    """brain 的 JSON 输出 -> 校验后的 List[UavOrder]（与 set_plan 内部同路）。"""
    tp = TeamPlan.from_llm_json(plan_dict, num_uavs)
    return tp.orders
