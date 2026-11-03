# -*- coding: utf-8 -*-
"""任务标签与任务指令（order）的规范定义。

上层是**集中式**规划：一次调用给出全队所有无人机的任务与目标（`TeamPlan`），
每架无人机的强化学习策略只看自己的那条 `UavOrder`，输出自己的 9 个离散动作之一。

本文件只定义"接口"，不含任何决策逻辑——训练期由课程采样器产生，
接入大模型后由 LLM Task Planner 产生、经 ReAd critic 打分通过后下发。

与第一版的区别：任务建议从"抽象偏好"（priority_front / relay_spacing）改成
**显式目标**。多火点、100x100 的任务区域下，"去哪"本身就是高层要决策的内容：
    - 感知任务：去探索哪片未知区域，或者绕哪个已知火点的边界巡逻（含环半径）
    - 中继任务：去覆盖哪一个救援人员
    - 返航补给：去哪个补给点（默认由环境做全局去冲突匹配）
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
import json
import math

# ---------------------------------------------------------------- 任务标签
TASK_FIRE_SENSING = 0   # 火情感知：搜寻未知火点 / 巡逻已知火线
TASK_COMM_RELAY = 1     # 通信中继：为地面救援人员提供覆盖并回传火势地图
TASK_RECHARGE = 2       # 返航补给：前往补给点
N_TASKS = 3

TASK_NAMES = {
    TASK_FIRE_SENSING: "fire_sensing",
    TASK_COMM_RELAY: "comm_relay",
    TASK_RECHARGE: "recharge",
}
TASK_FROM_NAME = {v: k for k, v in TASK_NAMES.items()}
TASK_FROM_NAME.update({"explore": TASK_FIRE_SENSING, "sensing": TASK_FIRE_SENSING,
                       "patrol": TASK_FIRE_SENSING, "relay": TASK_COMM_RELAY,
                       "comm": TASK_COMM_RELAY, "charge": TASK_RECHARGE})

# ------------------------------------------------------- 感知任务的两种模式
SENSE_EXPLORE = 0   # 搜寻：去指定的未知区域把地图铺开
SENSE_PATROL = 1    # 巡逻：绕指定火点的边界转，持续刷新火线状态
N_SENSE_MODES = 2

MODE_NAMES = {SENSE_EXPLORE: "explore", SENSE_PATROL: "patrol"}
MODE_FROM_NAME = {v: k for k, v in MODE_NAMES.items()}


@dataclass
class UavOrder:
    """高层下发给**单架**无人机的指令。全部字段都会被编码进该机的观测。

    task        : 0 感知 / 1 中继 / 2 补给
    mode        : 仅 task=0 有意义，SENSE_EXPLORE 去搜寻、SENSE_PATROL 绕火点巡逻
    target      : (x, y) 显式目标点。explore=未知区域中心，patrol=火点中心，
                  relay=指定人员位置（也可只给 rescuer_id，由环境查位置）。
                  None 表示"由你自己就近决定"，策略会退回到最近前沿/最近未覆盖人员。
    ring_radius : patrol 模式下希望保持的绕圈半径（格）。None = 自动按火点规模取
                  （等效半径 sqrt(size/pi) + 2），这是默认值——写死 6 格的话，
                  比 6 格还大的火场会让无人机贴着火里绕，白吃 w_in_fire 惩罚
    rescuer_id  : 中继任务指定覆盖哪个人员（1-based，与 Rescuer.rescuer_id 一致）
    proactive_handover : 是否要求提前交接中继角色（PRS-HRL 的 switching-aware）
    """
    task: int = TASK_FIRE_SENSING
    mode: int = SENSE_EXPLORE
    target: Optional[Tuple[float, float]] = None
    ring_radius: Optional[float] = None
    rescuer_id: Optional[int] = None
    proactive_handover: bool = True

    RING_SCALE = 20.0
    # mode one-hot(2) + target rel(3) + ring(1) + ring_err(1)
    # + has_explicit_target(1) + target_is_fire(1) + handover(1)
    VEC_DIM = 10

    @staticmethod
    def vector_dim():
        return UavOrder.VEC_DIM

    # 契约版本记录：v2 修复了混合字段（relay 同时给 target 与 rescuer_id）时
    # to_vector 优先用原 target、而 resolve_order_target 按 rescuer_id 优先解析
    # 造成的"观测编码目标 ≠ 奖励/评估实际目标"不一致（H200 训练计划 §3.2）。
    # v2 规则：调用方提供了有效 resolved_target 时优先编码它；没有解析值再用原 target。
    CONTRACT_VERSION = 2

    def to_vector(self, ux, uy, d_max, resolved_target=None, target_is_fire=False,
                  ring=None):
        """编码为 10 维向量。ux/uy 是本机位置，用于算目标的相对方向。

        resolved_target 允许调用方把 None 目标替换成"策略自己就近选的目标"，
        这样"有没有显式目标"这一位仍然如实反映高层是否指定过。
        CONTRACT_VERSION>=2：resolved_target 非空时优先于 self.target——
        观测中的目标必须与 resolve_order_target 实际解析出的目标一致。

        target_is_fire 区分"这个目标是火点（要绕着它转）"还是"这是一片待探索
        区域（要飞过去）"。少了这一位，patrol 指令在还没发现火点时观测上无法
        与 explore 区分，而奖励侧一旦发现火点就切到环半径势函数，两边打架。
        """
        vec = [0.0] * N_SENSE_MODES
        md = int(self.mode)
        if 0 <= md < N_SENSE_MODES:
            vec[md] = 1.0
        # 契约 v2：resolved_target 优先（与 resolve_order_target 的解析顺序一致）
        tgt = resolved_target if resolved_target is not None else self.target
        if tgt is None:
            vec += [0.0, 0.0, 1.0]
            dist = d_max
        else:
            dx, dy = float(tgt[0]) - float(ux), float(tgt[1]) - float(uy)
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                vec += [0.0, 0.0, 0.0]
            else:
                vec += [dx / dist, dy / dist, min(dist / d_max, 1.0)]
        rr = float(ring if ring is not None
                   else (self.ring_radius if self.ring_radius is not None else 6.0))
        vec.append(rr / self.RING_SCALE)
        # 环误差：正=离火点太远，负=离得太近。patrol 的核心控制量。
        vec.append(max(-1.0, min(1.0, (dist - rr) / self.RING_SCALE)))
        vec.append(1.0 if self.target is not None or self.rescuer_id is not None else 0.0)
        vec.append(1.0 if target_is_fire else 0.0)
        vec.append(1.0 if self.proactive_handover else 0.0)
        return vec

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["task"] = TASK_NAMES.get(int(self.task), "fire_sensing")
        d["mode"] = MODE_NAMES.get(int(self.mode), "explore")
        return d

    # ------------------------------------------------ 与 LLM 输出的对接
    @classmethod
    def from_llm_json(cls, obj) -> "UavOrder":
        """从 LLM 返回的单机指令 JSON 解析。

        容错策略与 A2C-LLM 的 AugParse 一致：字段缺失或非法一律回落默认值，
        不抛异常，由上层记录 violation flag。
        """
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:
                return cls()
        if not isinstance(obj, dict):
            return cls()
        o = cls()

        t = obj.get("task", obj.get("label"))
        if isinstance(t, str):
            o.task = TASK_FROM_NAME.get(t.strip().lower(), TASK_FIRE_SENSING)
            if t.strip().lower() == "patrol":
                o.mode = SENSE_PATROL
        elif isinstance(t, (int, float)) and 0 <= int(t) < N_TASKS:
            o.task = int(t)

        md = obj.get("mode")
        if isinstance(md, str):
            o.mode = MODE_FROM_NAME.get(md.strip().lower(), o.mode)
        elif isinstance(md, (int, float)) and 0 <= int(md) < N_SENSE_MODES:
            o.mode = int(md)

        tgt = obj.get("target", obj.get("goal", obj.get("target_pos")))
        if isinstance(tgt, dict):
            tgt = [tgt.get("x"), tgt.get("y")]
        if isinstance(tgt, (list, tuple)) and len(tgt) >= 2:
            try:
                o.target = (float(tgt[0]), float(tgt[1]))
            except (TypeError, ValueError):
                o.target = None

        rr = obj.get("ring_radius", obj.get("radius"))
        if isinstance(rr, (int, float)):
            o.ring_radius = float(min(max(rr, 2.0), 40.0))
        elif isinstance(rr, str) and rr.strip().lower() in ("auto", ""):
            o.ring_radius = None

        rid = obj.get("rescuer_id", obj.get("rescuer"))
        if isinstance(rid, (int, float)):
            o.rescuer_id = int(rid)

        ho = obj.get("proactive_handover", obj.get("handover"))
        if isinstance(ho, bool):
            o.proactive_handover = ho
        elif isinstance(ho, str):
            o.proactive_handover = ho.strip().lower() in ("true", "yes", "proactive", "1")
        return o


@dataclass
class TeamPlan:
    """一次集中式高层决策的完整输出：每架无人机一条 UavOrder。"""
    orders: List[UavOrder] = field(default_factory=list)

    def validate(self, num_uavs: int):
        """裁剪/补齐到 num_uavs，非法值回落默认指令。返回违规计数。"""
        violations = 0
        orders = list(self.orders)
        if len(orders) < num_uavs:
            violations += num_uavs - len(orders)
            orders += [UavOrder() for _ in range(num_uavs - len(orders))]
        elif len(orders) > num_uavs:
            violations += len(orders) - num_uavs
            orders = orders[:num_uavs]
        clean = []
        for o in orders:
            if not isinstance(o, UavOrder):
                o = UavOrder.from_llm_json(o)
                violations += 1
            if not (0 <= int(o.task) < N_TASKS):
                o.task = TASK_FIRE_SENSING
                violations += 1
            clean.append(o)
        self.orders = clean
        return violations

    @property
    def tasks(self):
        return [int(o.task) for o in self.orders]

    @classmethod
    def from_llm_json(cls, obj, num_uavs=None) -> "TeamPlan":
        """解析形如 {"uavs":[{...},{...}]} 或直接是一个 list 的 LLM 输出。"""
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:
                obj = []
        if isinstance(obj, dict):
            obj = obj.get("uavs", obj.get("orders", obj.get("plan", [])))
        if not isinstance(obj, (list, tuple)):
            obj = []
        plan = cls([UavOrder.from_llm_json(o) for o in obj])
        if num_uavs is not None:
            plan.validate(num_uavs)
        return plan

    def to_dict(self):
        return {"uavs": [o.to_dict() for o in self.orders]}
