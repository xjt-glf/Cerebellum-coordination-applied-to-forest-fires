"""部署补给协议：校验站点、提前下发充电任务，动作仍由 Actor 输出。"""
import copy
import math

from lowlevel.task_spec import TASK_RECHARGE, UavOrder


class ChargeProtocol:
    def __init__(self, wrapper, goals=None, reserve_steps=12):
        self.wrapper = wrapper
        self.goals = goals
        self.reserve_steps = reserve_steps
        self.targets = {}
        self.stranded = set()
        wrapper.env._deployment_nearest_charger = True
        wrapper.env._deployment_station_targets = {}

    def steps_to_pad(self, u, station):
        radius = int(getattr(self.wrapper.env, "charge_radius", 1))
        return max(0, math.ceil(max(abs(u.x - station[0]), abs(u.y - station[1])) - radius))

    def sync(self):
        env, e = self.wrapper, self.wrapper.env
        stations = [tuple(map(float, st)) for st in e.charging_stations]
        if not stations:
            return None, []
        orders = copy.deepcopy(env.orders)
        notes, targets = [], {}
        for i, u in enumerate(e.uavs):
            nearest = min(stations, key=lambda st: (
                self.steps_to_pad(u, st), math.hypot(st[0] - u.x, st[1] - u.y)))
            cost = e.energy_cost_per_step(i)
            required = cost * self.steps_to_pad(u, nearest)
            if u.e <= 0 and not e.at_station(u.x, u.y):
                if i not in self.stranded:
                    notes.append(f"[充电保护] UAV{i} 已在站外耗尽，无法自行移动；需要重新开局")
                    self.stranded.add(i)
            elif u.e <= required + cost * self.reserve_steps:
                if orders[i].task != TASK_RECHARGE:
                    orders[i] = UavOrder(task=TASK_RECHARGE)
                    notes.append(f"[充电保护] UAV{i} 电量 {u.e:.1f} 接近返航需求，提前补给")
            if orders[i].task != TASK_RECHARGE:
                continue
            requested = orders[i].target
            target = self.targets.get(i, nearest)
            if requested is not None:
                valid = [st for st in stations if max(abs(st[0]-requested[0]),
                                                       abs(st[1]-requested[1])) <= e.charge_radius]
                if valid:
                    target = min(valid, key=lambda st: self.steps_to_pad(u, st))
                else:
                    target = nearest
                    notes.append(f"[充电保护] UAV{i} 的 {requested} 不是补给点，改用 {target}")
            if cost * self.steps_to_pad(u, target) > u.e and required < cost * self.steps_to_pad(u, target):
                target = nearest
            targets[i] = target
            # 保持训练时的 charge 无显式目标编码；站点通过 assigned_station 统一解析。
            orders[i].target = None
            orders[i].rescuer_id = None
            if self.targets.get(i) != target:
                notes.append(f"[充电保护] UAV{i} 补给点 → {target}")
        changed = targets != self.targets or any(
            a.to_dict() != b.to_dict() for a, b in zip(orders, env.orders))
        self.targets = targets
        e._deployment_station_targets = targets
        if not changed:
            return None, notes
        e._obs_cache = None
        obs = env.set_plan(orders)
        if self.goals is not None:
            self.goals.current = copy.deepcopy(orders)
            self.goals.manager.sync(e, orders, env.t)
        return obs, notes
