"""Explicit personnel tasks over the existing movement/extinguishing executor.

No actor actions, fire physics or communication rules are changed. Commands wait
for the existing geometric communication flag, then own their selected people
until replaced/released. Other people retain the legacy automatic behaviour.
"""
import copy
import math


class PersonnelController:
    TASKS = {"move", "contain", "hold", "retreat", "auto"}
    SECTORS = {"any", "north", "south", "east", "west"}

    def __init__(self, env):
        self.env = env
        self.pending = {}
        self.active = {}
        self.events = []
        self.serial = 0

    def normalize(self, orders):
        if not isinstance(orders, list):
            raise ValueError("orders 必须是列表")
        ids = {r.rescuer_id for r in self.env.rescuers}
        seen, result = set(), []
        H, W = self.env.grid_size

        def point(v):
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise ValueError("坐标必须是 [x,y]")
            if any(isinstance(q, bool) for q in v):
                raise ValueError("坐标必须是有限数值")
            x, y = map(float, v)
            if not math.isfinite(x + y) or not (.5 <= x <= H-1.5 and .5 <= y <= W-1.5):
                raise ValueError("人员坐标必须在可行走边界内")
            return [x, y]

        for raw in orders:
            if not isinstance(raw, dict):
                raise ValueError("每条指令必须是对象")
            task = raw.get("task")
            if task == "transit":
                task = "move"
            if task not in self.TASKS:
                raise ValueError("task 必须是 move/contain/hold/retreat/auto")
            people = raw.get("person_ids", [raw.get("person_id")])
            if not isinstance(people, list) or not people:
                raise ValueError("请指定 person_ids")
            for rid in people:
                if isinstance(rid, bool) or not isinstance(rid, int) or rid not in ids:
                    raise ValueError(f"人员编号不存在：{rid}；合法编号 {sorted(ids)}")
                if rid in seen:
                    raise ValueError(f"人员 {rid} 同批收到多条指令")
                seen.add(rid)
            target = point(raw["target"]) if raw.get("target") is not None else None
            route = raw.get("waypoints", [])
            if not isinstance(route, list) or len(route) > 8:
                raise ValueError("waypoints 必须为最多8个航点")
            route = [point(v) for v in route]
            if task in {"move", "retreat", "contain"} and target is None and not route:
                raise ValueError(f"{task} 需要 target 或 waypoints")
            if not route and target is not None:
                route = [target]
            sector = raw.get("approach_sector", "any")
            if sector not in self.SECTORS:
                raise ValueError("approach_sector 必须为 any/north/south/east/west")
            ttl = raw.get("valid_for_ticks", 300)
            if isinstance(ttl, bool) or not isinstance(ttl, int) or not 1 <= ttl <= 3000:
                raise ValueError("valid_for_ticks 必须为1至3000的整数")
            result.append(dict(person_ids=people[:], task=task, target=target,
                               waypoints=route, approach_sector=sector,
                               valid_for_ticks=ttl))
        return result

    def event(self, tick, rid, kind, **extra):
        self.events.append(dict(t=tick, person_id=rid, event=kind, **extra))

    def enqueue(self, orders, tick, source="operator", reason=""):
        orders = self.normalize(orders)  # validate entire batch before mutation
        for raw in orders:
            for rid in raw["person_ids"]:
                self.serial += 1
                cmd = copy.deepcopy(raw)
                cmd.update(order_id=f"p-{tick}-{self.serial}", issued_tick=tick,
                           expires=tick+raw["valid_for_ticks"], source=source,
                           reason=reason, index=0, visited=[], state="pending",
                           last_xy=None, stagnant=0)
                self.pending[rid] = cmd
                self.event(tick, rid, "issued", task=cmd["task"],
                           source=source, order_id=cmd["order_id"], reason=reason)

    def prepare(self, tick):
        """Called AFTER legacy coverage update, before automatic assignment."""
        for person in self.env.rescuers:
            rid = person.rescuer_id
            cmd = self.pending.get(rid)
            if cmd is not None:
                if tick >= cmd["expires"]:
                    self.event(tick, rid, "delivery_expired", order_id=cmd["order_id"])
                    del self.pending[rid]
                elif person.is_covered:
                    del self.pending[rid]
                    if cmd["task"] == "auto":
                        self.active.pop(rid, None)
                        person.target_pos, person.status = None, "idle"
                        person.cluster_anchor = None
                        self.event(tick, rid, "released", source=cmd["source"])
                    else:
                        cmd["state"] = "active"
                        cmd["received_tick"] = tick
                        if not cmd["waypoints"]:
                            cmd["waypoints"] = [[float(person.x), float(person.y)]]
                        self.active[rid] = cmd
                        self.event(tick, rid, "activated", task=cmd["task"],
                                   order_id=cmd["order_id"], source=cmd["source"])
                    person.cached_path = person.cached_path_target = None
            cmd = self.active.get(rid)
            if cmd and tick >= cmd["expires"] and cmd["state"] != "expired":
                cmd["state"] = "expired"
                self.event(tick, rid, "expired", order_id=cmd["order_id"])
        return set(self.active)

    def _safe_goal(self, goal):
        """Choose a known unburned alternative when the requested goal burns.

        Navigation itself retains the legacy BFS's map assumptions in this
        incremental version. No new wireless or perception model is introduced.
        """
        e = self.env
        x, y = map(lambda v: int(round(v)), goal)
        if not e.is_blocked_for_people(x, y):
            return tuple(goal)
        M = e.rescuers_map_info
        candidates = []
        for xx in range(max(1, x-5), min(e.grid_size[0]-1, x+6)):
            for yy in range(max(1, y-5), min(e.grid_size[1]-1, y+6)):
                if M[xx, yy] == 0 and not e.is_blocked_for_people(xx, yy):
                    candidates.append((xx, yy))
        return min(candidates, key=lambda q: math.hypot(q[0]-x, q[1]-y)) if candidates else None

    def _contain_goal(self, person, cmd):
        e = self.env
        anchor = cmd["target"] or cmd["waypoints"][-1]
        clusters = e.known_fire_clusters(map_info=e.rescuers_map_info)
        if not clusters:
            return None, "completed" if cmd.get("seen_region") else "waiting_information"
        cluster = min(clusters, key=lambda c: math.hypot(c["centroid"][0]-anchor[0], c["centroid"][1]-anchor[1]))
        if math.hypot(cluster["centroid"][0]-anchor[0], cluster["centroid"][1]-anchor[1]) > 20:
            return None, "completed" if cmd.get("seen_region") else "waiting_information"
        cmd["seen_region"] = True
        if cluster.get("contained"):
            return None, "completed"
        cx, cy = cluster["centroid"]
        sector = cmd["approach_sector"]
        def inside(q):
            dx, dy = q[0]-cx, q[1]-cy
            return (sector == "any" or sector == "west" and dx <= 0 and abs(dx) >= abs(dy)
                    or sector == "east" and dx >= 0 and abs(dx) >= abs(dy)
                    or sector == "north" and dy >= 0 and abs(dy) >= abs(dx)
                    or sector == "south" and dy <= 0 and abs(dy) >= abs(dx))
        candidates = [q for q in cluster["approach"] if inside(q) and not e.is_blocked_for_people(*q)]
        if not candidates:
            # A spreading front can invalidate all remembered work positions.
            # Keep the assigned region/sector, but find a nearby safe approach
            # instead of falling back to another fire or waiting forever.
            old_ring = sorted((q for q in cluster["approach"] if inside(q)),
                              key=lambda q: math.hypot(q[0]-person.x, q[1]-person.y))[:12]
            candidates = [p for q in old_ring if (p := self._safe_goal(q)) is not None and inside(p)]
        if not candidates:
            return None, "blocked"
        goal = min(candidates, key=lambda q: math.hypot(q[0]-person.x, q[1]-person.y))
        return goal, "working" if math.hypot(goal[0]-person.x, goal[1]-person.y) < .75 else "active"

    def before_move(self, tick):
        def hold(person):
            # Keep the existing local escape branch reachable even while holding.
            if self.env.is_blocked_for_people(int(round(person.x)), int(round(person.y))):
                person.target_pos, person.status = (person.x, person.y), "commanded"
            else:
                person.target_pos, person.status = None, "idle"
        for person in self.env.rescuers:
            cmd = self.active.get(person.rescuer_id)
            if cmd is None:
                continue
            if cmd["state"] == "expired":
                hold(person)
                continue
            if cmd["task"] == "contain":
                goal, state = self._contain_goal(person, cmd)
            else:
                points = cmd["waypoints"]
                goal = points[cmd["index"]]
                if math.hypot(person.x-goal[0], person.y-goal[1]) < .75:
                    if cmd["index"] + 1 < len(points):
                        cmd["index"] += 1
                        goal = points[cmd["index"]]
                        self.event(tick, person.rescuer_id, "waypoint_reached", index=cmd["index"]-1)
                    else:
                        if cmd["state"] not in {"completed", "holding"}:
                            self.event(tick, person.rescuer_id, "completed", order_id=cmd["order_id"])
                        cmd["state"] = "holding" if cmd["task"] == "hold" else "completed"
                        hold(person)
                        continue
                goal, state = self._safe_goal(goal), "active"
                if goal is None:
                    state = "blocked"
            cmd["state"] = state
            if goal != person.target_pos:
                person.cached_path = person.cached_path_target = None
            person.target_pos = goal
            person.status = "commanded" if goal is not None else "idle"
            if goal is None:
                hold(person)
            cmd["resolved_target"] = list(goal) if goal is not None else None

    def after_move(self, tick):
        for person in self.env.rescuers:
            cmd = self.active.get(person.rescuer_id)
            if not cmd:
                continue
            xy = (float(person.x), float(person.y))
            if cmd["last_xy"] and math.dist(xy, cmd["last_xy"]) < 1e-7 and person.target_pos:
                cmd["stagnant"] += 1
                if cmd["stagnant"] == 20:
                    self.event(tick, person.rescuer_id, "no_progress", target=person.target_pos)
                if cmd["stagnant"] >= 20:
                    cmd["state"] = "blocked"
            else:
                cmd["stagnant"] = 0
            cmd["last_xy"] = xy

    def person_frame(self, person):
        cmd = self.active.get(person.rescuer_id)
        pending = self.pending.get(person.rescuer_id)
        return dict(id=person.rescuer_id, x=round(float(person.x), 2), y=round(float(person.y), 2),
                    c=bool(person.is_covered), aoi=int(person.AoI),
                    task=cmd["task"] if cmd else "auto", state=cmd["state"] if cmd else "auto",
                    target=list(person.target_pos) if person.target_pos is not None else None,
                    waypoints=copy.deepcopy(cmd["waypoints"][cmd["index"]:]) if cmd else [],
                    source=cmd["source"] if cmd else "heuristic",
                    pending=copy.deepcopy(pending) if pending else None,
                    order_id=cmd["order_id"] if cmd else None)

    def snapshot(self, tick):
        return dict(t=tick, communication="existing_geometric_link; no rate/queue model",
                    people=[self.person_frame(r) for r in self.env.rescuers],
                    known_fire_clusters=[dict(center=list(c["centroid"]), size=int(c["size"]),
                                              contained=bool(c["contained"]), approach=[list(q) for q in c["approach"][:24]])
                                         for c in self.env.known_fire_clusters(map_info=self.env.rescuers_map_info)],
                    events=copy.deepcopy(self.events[-60:]))
