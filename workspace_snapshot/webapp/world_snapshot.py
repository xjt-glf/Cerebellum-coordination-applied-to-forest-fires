"""给大模型的逐格环境快照；真值模式用于观察充分信息下的决策能力。"""
import math

import numpy as np

from lowlevel.obs_builder import assigned_station, resolve_order_target, shared_cache


def fire_rows(matrix):
    """所有格子逐行编码，行号为x、行内位置为y。"""
    a = np.asarray(matrix)
    chars = np.full(a.shape, "?", dtype="<U1")
    chars[a == 0] = "."
    chars[(a > 0) & (a < 2)] = "f"
    chars[a == 1] = "F"
    chars[a == 2] = "b"
    chars[a == 3] = "e"
    return {"shape": list(a.shape), "rows_x": ["".join(row) for row in chars]}


def numeric_runs(matrix, decimals=None):
    """每行[value,count]游程；完整保留每个格子，火势数值不舍入。"""
    a = np.asarray(matrix)
    rows = []
    for row in a:
        values = [round(float(v), decimals) if decimals is not None else float(v) for v in row]
        runs = []
        for value in values:
            if runs and runs[-1][0] == value:
                runs[-1][1] += 1
            else:
                runs.append([value, 1])
        rows.append(runs)
    return {"shape": list(a.shape), "rows_x_value_count": rows,
            "decimal_places": decimals}


def full_snapshot(wrapper, goals=None, recent_decisions=()):
    e = wrapper.env
    cache = shared_cache(e)
    maps = {
        "true_fire": fire_rows(e.grid["burning_state"]),
        "true_burning_values": numeric_runs(e.grid["burning_state"]),
        "team_known_fire": fire_rows(e.team_map_info),
        "personnel_shared_known_fire": fire_rows(e.rescuers_map_info),
        "terrain_height": numeric_runs(e.grid["height"], decimals=3),
        "flammability": numeric_runs(e.grid["flammability"], decimals=3),
    }
    uavs = []
    for i, u in enumerate(e.uavs):
        order = wrapper.orders[i]
        target, is_fire, ring = resolve_order_target(e, i, order, cache)
        station, distance = assigned_station(e, i)
        goal = goals.manager.goals.get(i) if goals is not None else None
        progress = None
        if goal is not None:
            progress = dict(kind=goal.kind, created_t=goal.created_t,
                            deadline_t=goal.deadline_t, arrived=goal.arrived,
                            completed=goal.completed, arrival_t=goal.arrival_t,
                            covered_angular_sectors=sorted(goal.sectors))
            if goal.initial_unknown is not None:
                denominator = max(1, int(goal.initial_unknown.sum()))
                progress["own_region_completion"] = float(
                    (goal.own_seen & goal.initial_unknown).sum()) / denominator
        maps[f"uav_{i}_known_fire"] = fire_rows(u.map_info)
        uavs.append(dict(
            id=i, position=[float(u.x), float(u.y)], energy=float(u.e),
            max_energy=float(u.e_max), water_or_payload=float(u.d),
            current_order=order.to_dict(), resolved_target=target,
            distance_to_target=(math.hypot(u.x-target[0], u.y-target[1]) if target else None),
            target_is_fire=is_fire, actual_patrol_radius=ring,
            selected_charger=station, distance_to_charger=distance,
            energy_per_step=e.energy_cost_per_step(i), at_charger=e.at_station(u.x, u.y),
            recharged=bool(u.recharged), depleted=bool(u.e <= 0),
            blocked=bool(u.blocked), out_of_bounds=bool(u.oob),
            goal_progress=progress,
            personnel_in_own_link=[int(r.rescuer_id) for r in e.rescuers
                                  if math.hypot(u.x-r.x, u.y-r.y) <= e.uav_comm_radius]))
    people = [dict(id=int(r.rescuer_id), position=[float(r.x), float(r.y)],
                   status=r.status, target=r.target_pos, covered=bool(r.is_covered),
                   aoi=float(r.AoI), fire_cluster_anchor=r.cluster_anchor,
                   remaining_path=r.cached_path,
                   uav_links=[i for i, u in enumerate(e.uavs)
                              if math.hypot(u.x-r.x, u.y-r.y) <= e.uav_comm_radius])
              for r in e.rescuers]
    clusters = [dict(centroid=list(cl["centroid"]), size=int(cl["size"]),
                     contained=bool(cl.get("contained", False)),
                     boundary_cells=[list(cell) for cell in cl["boundary"]])
                for cl in e.known_fire_clusters(map_info=e.grid["burning_state"])]
    return dict(
        mode="full_environment_truth", time=wrapper.t, max_steps=wrapper.max_steps,
        grid_size=list(e.grid_size),
        encoding={"coordinates": "所有矩阵都是matrix[x,y]；第x行的第y位，不是通常图像的行列约定。",
                  "fire_symbols": {"?": "未知", ".": "未过火/健康", "f": "燃烧中",
                                   "F": "完全燃烧（数值1）", "b": "烧尽（2）", "e": "已扑灭（3）"},
                  "numeric_runs": "每行按y从0递增，依次展开[value,count]；每行完整覆盖W格，无格子省略。",
                  "numeric_precision": "火势数值不舍入；地形高度和可燃系数逐格保留3位小数。"},
        rules=dict(energy_per_step=float(e.energy_cost), sensing_radius=e.sense_radius,
                   uav_communication_radius=e.uav_comm_radius,
                   base_position=list(e.base_pos), base_communication_radius=e.base_comm_radius,
                   charging_stations=[list(st) for st in e.charging_stations],
                   charging_pad_radius=e.charge_radius, rescuer_speed=e.rescuer_speed,
                   extinguish_radius=e.extinguish_radius,
                   extinguish_rate=float(e.extinguish_rate),
                   spread_gain=e.spread_gain, wind_speed=e.wind_speed,
                   wind_direction_radians=e.wind_direction),
        maps=maps, uavs=uavs, personnel=people, true_fire_clusters=clusters,
        recent_decisions=[dict(t=row["t"], applied_plan=row["plan"],
                               metrics=row.get("brief", {}), source=row["meta"].get("source"))
                          for row in recent_decisions[-5:]])
