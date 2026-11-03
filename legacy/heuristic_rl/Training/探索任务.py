import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Env.低层环境 import GridEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import torch.optim as optim
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dirs8 = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]  # 与 action_set 前8个一致
def _in_bounds(x,y,H,W): return 0<=x<H and 0<=y<W

def _is_boundary(M, x, y):
    # “只沿边”判定：该格为火(0<v<2) 且 4邻至少一个健康格(=0)
    v = M[x,y]
    if not (v>0.0 and v<2.0): return False
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx,ny = x+dx,y+dy
        if _in_bounds(nx,ny,*M.shape) and M[nx,ny]==0.0:
            return True
    return False

def _edge_union(uav):
    # 并集并去重
    S = set()
    S |= set(getattr(uav,"B",[]))
    S |= set(getattr(uav,"B_flank",[]))
    S |= set(getattr(uav,"B_tail",[]))
    return S

def step_communicate_nearest(uav, env, dist_stop: float = 3.0) -> int:
    """
    通信子任务启发式：
    - 找到距离自己最近的无人机；
    - 若最近距离 < dist_stop，则原地不动（便于持续通信）；
    - 否则，朝该无人机迈出一步（8邻域），返回动作索引 0..8。
    """
    # 无其他无人机，直接原地
    if not hasattr(env, "uavs") or len(env.uavs) <= 1:
        return 8

    # 收集所有无人机位置（用索引区分自身）
    positions = [(i, int(v.x), int(v.y)) for i, v in enumerate(env.uavs)]
    # 找到自身索引
    self_idx = None
    for i, v in enumerate(env.uavs):
        if v is uav:
            self_idx = i
            break

    ux, uy = int(uav.x), int(uav.y)
    min_dist = float("inf")
    target_xy = None
    for i, x, y in positions:
        if i == self_idx:
            continue
        d = abs(x - ux) + abs(y - uy)  # 曼哈顿距离
        if d < min_dist:
            min_dist = d
            target_xy = (x, y)

    # 没有队友，或距离已经足够近 → 原地等待
    if target_xy is None or min_dist < dist_stop:
        return 8

    tx, ty = target_xy
    dx = np.sign(tx - ux)
    dy = np.sign(ty - uy)
    move = (int(dx), int(dy))

    action_set = [
        np.array([1,0]),  np.array([1,1]),  np.array([0,1]),
        np.array([-1,1]), np.array([-1,0]), np.array([-1,-1]),
        np.array([0,-1]), np.array([1,-1]), np.array([0,0])
    ]
    for idx, a in enumerate(action_set):
        if (a[0] == move[0]) and (a[1] == move[1]):
            return idx
    return 8

def heuristic_actions(obs_list):
    """
    输入:
        obs_list: List of numpy arrays，每个obs对应一架无人机
    输出:
        actions: List of int，每个动作索引
    """
    action_set = [
        np.array([1,0]), np.array([1,1]), np.array([0,1]),
        np.array([-1,1]), np.array([-1,0]), np.array([-1,-1]),
        np.array([0,-1]), np.array([1,-1]), np.array([0,0])
    ]

    action_indices = []

    for obs in obs_list:
        memory = obs[0]
        local_fire = np.zeros((3,3), dtype=np.float32)
        local_fire[0,0] = obs[1]
        local_fire[0,1] = obs[2]
        local_fire[0,2] = obs[3]
        local_fire[1,0] = obs[4]
        local_fire[1,1] = obs[5]
        local_fire[1,2] = obs[6]
        local_fire[2,0] = obs[7]
        local_fire[2,1] = obs[8]
        local_fire[2,2] = obs[9]

        v_unit = obs[10:12]
        e_k = obs[12]
        w_k = obs[13:15]
        pos = obs[15:17]
        ammo = obs[17]

        flag =False
        # print(obs)
        if ammo < 0.01:
            flag = True


        if memory == 0.0:
            # 第一阶段: 朝火中心
            fire_center = np.array([25,25])
            to_fire = fire_center - pos*50
            to_fire_unit = to_fire / (np.linalg.norm(to_fire)+1e-8)
            best_u = min(
                action_set,
                key=lambda u: np.linalg.norm(
                    (u/(np.linalg.norm(u)+1e-8)) - to_fire_unit
                )
            )
        else:
            # 第二阶段: 绕圈
            # 先选基准巡逻动作
            v_u = min(
                action_set,
                key=lambda u: np.linalg.norm(
                    (u / (np.linalg.norm(u)+1e-8)) - v_unit
                )
            )

            # 基于巡逻动作生成方向
            left1 = np.array([-v_u[1], v_u[0]])
            i, j = v_u
            numerator = np.array([i-j, i+j])
            denom = max(abs(i+j), abs(i-j),1.0)
            left2 = numerator / denom
            right = np.array([v_u[1], -v_u[0]])

            # 中心坐标
            cx, cy = 1,1
            lx1 = cx + int(round(left1[0]))
            ly1 = cy + int(round(left1[1]))
            lx2 = cx + int(round(left2[0]))
            ly2 = cy + int(round(left2[1]))

            left1_fire = False
            left2_fire = False

            if 0<=lx1<3 and 0<=ly1<3:
                v = local_fire[lx1,ly1]
                if (0<v<=1.0) or np.isclose(v,3.0):
                    left1_fire = True

            if 0<=lx2<3 and 0<=ly2<3:
                v = local_fire[lx2,ly2]
                if (0<v<=1.0) or np.isclose(v,3.0):
                    left2_fire = True

            if left1_fire:
                # print(f"左1格子有火,旋转向量方向{v_u},自身位置{pos},坐标为{pos+left1}")
                best_u = min(
                    action_set,
                    key=lambda u: np.linalg.norm(
                        (u/(np.linalg.norm(u)+1e-8)) - left1
                    )
                )
            elif left2_fire:
                # print(f"左2格子有火,旋转向量方向{v_u},自身位置{pos},坐标为{pos+left2}")
                best_u = min(
                    action_set,
                    key=lambda u: np.linalg.norm(
                        (u/(np.linalg.norm(u)+1e-8)) - left2
                    )
                )
            else:
                # 默认选巡逻方向
                best_u = v_u
                # 检查局部火图未着火格子数量
                if np.sum(local_fire==0) >=6:
                    # 改为右方向
                    # print("火点太少，改为右方向")
                    best_u = min(
                        action_set,
                        key=lambda u: np.linalg.norm(
                            (u/(np.linalg.norm(u)+1e-8)) - right
                        )
                    )

        #换弹机制
        if flag ==True:
            reload_point = np.array([5,5])
            to_reload = reload_point - (pos * 50)
            to_reload_unit = to_reload / (np.linalg.norm(to_reload)+1e-8)
            best_u = min(
                action_set,
                key=lambda u: np.linalg.norm((u / (np.linalg.norm(u)+1e-8)) - to_reload_unit)
            )
            
            # ✅ 立即return该动作对应的索引
            for idx, a in enumerate(action_set):
                if np.allclose(a, best_u):
                    action_indices.append(idx)

        # 避障逻辑
        elif e_k==1.0 and np.linalg.norm(w_k+best_u)<=1 and flag==False:
            action_indices.append(8)  # 停下
        #排除换弹和避障
        else:
            found = False
            for idx, a in enumerate(action_set):
                if np.allclose(a, best_u):
                    action_indices.append(idx)
                    found = True
                    break
            if not found:
                raise ValueError("Best action not found in action_set!")

    return action_indices

def heuristic_action(obs,uav):
    """
    输入:
        obs: 单个numpy array
    输出:
        action_idx: int
    """
    action_set = [
        np.array([1,0]), np.array([1,1]), np.array([0,1]),
        np.array([-1,1]), np.array([-1,0]), np.array([-1,-1]),
        np.array([0,-1]), np.array([1,-1]), np.array([0,0])
    ]

    memory = obs[0]

    # 重建局部火情
    local_fire = np.zeros((3,3), dtype=np.float32)
    local_fire[0,0] = obs[1]
    local_fire[0,1] = obs[2]
    local_fire[0,2] = obs[3]
    local_fire[1,0] = obs[4]
    local_fire[1,1] = obs[5]
    local_fire[1,2] = obs[6]
    local_fire[2,0] = obs[7]
    local_fire[2,1] = obs[8]
    local_fire[2,2] = obs[9]

    v_unit = obs[10:12]
    e_k = obs[12]
    w_k = obs[13:15]
    pos = obs[15:17]
     # === Rescue机制监测 ===
    head_len, flank_len, tail_len = len(uav.B), len(uav.B_flank), len(uav.B_tail)
    stuck_sum = head_len + flank_len + tail_len

    # 用静态字典记录每个uav的历史状态
    if not hasattr(heuristic_action, "_rescue_hist"):
        heuristic_action._rescue_hist = {}
    key = uav.uav_id  # 假设每个UAV有唯一ID

    prev = heuristic_action._rescue_hist.get(key, {"best": float("inf"), "cnt": 0})
    if stuck_sum < prev["best"]:
        prev["best"] = stuck_sum
        prev["cnt"] = 0
    else:
        prev["cnt"] += 1
    heuristic_action._rescue_hist[key] = prev
    if not hasattr(heuristic_action, "_rescue_burst"):
        heuristic_action._rescue_burst = {}   # {uav_id: remaining_steps}
    if memory == 0.0:
        # 第一阶段: 朝火中心
        fire_center = np.array([15,25])
        to_fire = fire_center - (pos*50)
        # print(to_fire)
        to_fire_unit = to_fire / (np.linalg.norm(to_fire)+1e-8)
        best_u = min(
            action_set,
            key=lambda u: np.linalg.norm(
                (u/(np.linalg.norm(u)+1e-8)) - to_fire_unit
            )
        )
    else:
        # ===== 若正在进行两步冲刺：优先执行冲刺步 =====
        remain = heuristic_action._rescue_burst.get(uav.uav_id, 0)
        if remain > 0:
            candidates = []
            for (fx, fy) in list(uav.B) + list(uav.B_flank) + list(uav.B_tail):
                dx, dy = fx - int(uav.x), fy - int(uav.y)
                dist2 = dx*dx + dy*dy
                candidates.append(((dx, dy), dist2))
            if candidates:
                (dx, dy), _ = min(candidates, key=lambda t: t[1])
                to_fire_unit = np.array([dx, dy], dtype=float)
                if not (dx == 0 and dy == 0):
                    to_fire_unit /= (np.linalg.norm(to_fire_unit) + 1e-8)
                    best_u = min(
                        action_set,
                        key=lambda u: np.linalg.norm(
                            (u/(np.linalg.norm(u)+1e-8)) - to_fire_unit
                        )
                    )
                    # 扣减一次冲刺剩余步数并返回
                    heuristic_action._rescue_burst[uav.uav_id] = remain - 1
                    
                    for idx, a in enumerate(action_set):
                        if np.allclose(a, best_u):
                            return idx
        # print(prev["cnt"])
        if prev["cnt"] >= 10:
            # 开启两步冲刺
            heuristic_action._rescue_burst[uav.uav_id] = 2

            candidates = []
            for (fx, fy) in list(uav.B) + list(uav.B_flank) + list(uav.B_tail):
                dx, dy = fx - int(uav.x), fy - int(uav.y)
                dist2 = dx*dx + dy*dy
                candidates.append(((dx, dy), dist2))
            if candidates:
                (dx, dy), _ = min(candidates, key=lambda t: t[1])
                to_fire_unit = np.array([dx, dy], dtype=float)
                if not (dx == 0 and dy == 0):
                    to_fire_unit /= (np.linalg.norm(to_fire_unit) + 1e-8)
                    best_u = min(
                        action_set,
                        key=lambda u: np.linalg.norm(
                            (u/(np.linalg.norm(u)+1e-8)) - to_fire_unit
                        )
                    )
                    # 重置无进展计数，并执行第1步；将剩余步数设为1
                    prev["cnt"] = 0
                    heuristic_action._rescue_hist[key] = prev
                    heuristic_action._rescue_burst[uav.uav_id] = 1
                    if e_k == 2.0 and np.linalg.norm(w_k + best_u) <= 1:
                        return 8
                    for idx, a in enumerate(action_set):
                        if np.allclose(a, best_u):
                            return idx
        # 第二阶段: 巡逻
        v_u = min(
            action_set,
            key=lambda u: np.linalg.norm(
                (u / (np.linalg.norm(u)+1e-8)) - v_unit
            )
        )

        left1 = np.array([-v_u[1], v_u[0]])
        i, j = v_u
        numerator = np.array([i-j, i+j])
        denom = max(abs(i+j), abs(i-j),1.0)
        left2 = numerator / denom
        right = np.array([v_u[1], -v_u[0]])

        cx, cy = 1,1
        lx1 = cx + int(round(left1[0]))
        ly1 = cy + int(round(left1[1]))
        lx2 = cx + int(round(left2[0]))
        ly2 = cy + int(round(left2[1]))

        left1_fire = False
        left2_fire = False

        if 0<=lx1<3 and 0<=ly1<3:
            v = local_fire[lx1,ly1]
            if (0<v<=1.0) or np.isclose(v,3.0):
                left1_fire = True

        if 0<=lx2<3 and 0<=ly2<3:
            v = local_fire[lx2,ly2]
            if (0<v<=1.0) or np.isclose(v,3.0):
                left2_fire = True

        if left1_fire:
            best_u = min(
                action_set,
                key=lambda u: np.linalg.norm(
                    (u/(np.linalg.norm(u)+1e-8)) - left1
                )
            )
        elif left2_fire:
            best_u = min(
                action_set,
                key=lambda u: np.linalg.norm(
                    (u/(np.linalg.norm(u)+1e-8)) - left2
                )
            )
        else:
            best_u = v_u
            if np.sum(local_fire==0) >=6:
                best_u = min(
                    action_set,
                    key=lambda u: np.linalg.norm(
                        (u/(np.linalg.norm(u)+1e-8)) - right
                    )
                )


    if e_k==2.0 and np.linalg.norm(w_k+best_u)<=1:
        return 8  # 停下
    else:
    # 安全找索引
        for idx, a in enumerate(action_set):
            if np.allclose(a, best_u):
                return idx

    raise ValueError("Best action not found in action_set!")

def step_go_charge(uav, env):
    """
    充电启发式：面向最近充电站，取与目标方向最接近的 8-邻域动作（或原地）。
    返回动作索引 0..8
    """
    action_set = [
        np.array([1,0]),  np.array([1,1]),  np.array([0,1]),
        np.array([-1,1]), np.array([-1,0]), np.array([-1,-1]),
        np.array([0,-1]), np.array([1,-1]), np.array([0,0])
    ]
    # 最近充电站
    cx, cy = env.nearest_charger((int(uav.x), int(uav.y)))
    dx, dy = cx - int(uav.x), cy - int(uav.y)

    # 已在充电站：原地
    if dx == 0 and dy == 0:
        return 8

    # 归一化到最近离散方向
    vec = np.array([np.sign(dx), np.sign(dy)], dtype=float)
    best_action = min(
        range(len(action_set)),
        key=lambda i: np.linalg.norm(
            (action_set[i] / (np.linalg.norm(action_set[i]) + 1e-8)) -
            (vec         / (np.linalg.norm(vec)         + 1e-8))
        )
    )
    return best_action

def has_healthy_neighbor(x, y, M, W, H):
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H:
            if M[nx, ny] == 0:  # 健康格
                return True
    return False

def select_band_and_boundary(uav, k=1, wind=(1,0)):
    M = uav.map_info
    H, W = M.shape
    # 下风端“最前”投影（此处简化：wind=(1,0) → 取 x 最大）
    xs, ys = np.where((M > 0) & (M < 2))
    if xs.size == 0:
        return []
    x1 = xs.max()
    band_mask = (np.arange(W)[:, None] >= x1 - k)  # 注意坐标轴，你用的是 M[x,y]
    B = []
    for x, y in zip(xs, ys):
        if x < x1 - k: 
            continue
        # 边界判据：四邻至少一个非燃/未知
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if has_healthy_neighbor(x, y, M, W, H):
                B.append((x,y))
                break
    uav.B=B   
    return B  # 边界候选
import numpy as np

def select_flank_by_ybands(uav, k=1):
    """
    侧翼边界点选取：
    - 基于 y_max / y_min 回推 k 格，构造左右两条“侧翼带”
    - 在带内选择边界点（四邻至少一个健康格）
    - 自动排除 uav.B 中已有的点（避免跟火头等已有边界混淆）
    
    写回：
      uav.B_flank = [(x,y), ...]
    返回：
      同上列表
    """
    M = uav.map_info
    H, W = M.shape

    xs, ys = np.where((M > 0) & (M < 2))
    if xs.size == 0:
        uav.B_flank = []
        return []

    y_max = ys.max()
    y_min = ys.min()

    B_flank = []
    for x, y in zip(xs, ys):
        # 在左右两条带内
        if not (y >= y_max - k or y <= y_min + k):
            continue

        # 排除已有的 uav.B
        if hasattr(uav, "B") and (x, y) in uav.B:
            continue

        # 边界判据：四邻至少一个健康格
        has_healthy = False
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < H and 0 <= ny < W and M[nx, ny] == 0:
                has_healthy = True
                break
        if has_healthy:
            B_flank.append((x, y))

    uav.B_flank = B_flank
    return B_flank
import numpy as np

def select_tail_by_xband(uav, k=1):
    """
    尾部边界点选取（仿照 x_max 回推的逻辑，但改为 x_min 前推 k 格）：
    - 找到燃烧格 (0 < M < 2) 的 x_min
    - 构造“尾部带”：x <= x_min + k
    - 在带内筛“边界点”（四邻至少一个健康格 M==0）
    - 排除 uav.B 与 uav.B_flank 中已有的点

    写回：
      uav.B_tail = [(x,y), ...]
    返回：
      同上列表
    """
    M = uav.map_info
    H, W = M.shape

    xs, ys = np.where((M > 0) & (M < 2))
    if xs.size == 0:
        uav.B_tail = []
        return []

    x_min = xs.min()
    B_tail = []

    for x, y in zip(xs, ys):
        # 在尾部带内：x <= x_min + k
        if x > x_min + k:
            continue

        # 排除已有的边界集合
        if hasattr(uav, "B") and (x, y) in uav.B:
            continue
        if hasattr(uav, "B_flank") and (x, y) in getattr(uav, "B_flank", []):
            continue

        # 边界判据：四邻至少一个健康格
        has_healthy = False
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and M[nx, ny] == 0:
                has_healthy = True
                break
        if has_healthy:
            B_tail.append((x, y))

    uav.B_tail = B_tail
    return B_tail


import numpy as np

def step_attack_head_minimal(uav, burning_state,env,
                             weights=(0, 1, 0)):
    """
    仅用火势 + 距离 + 健康邻居比例 计算优先级，保存到 uav.bp（与 uav.B 对齐），
    选择最高优先级点为目标，并返回朝目标前进的一步动作索引（0..8）。
    - uav.B: list[(x,y)] 边界点（已由你提前填好）
    - burning_state: env.grid['burning_state']，用索引 [x, y]
    - weights: (alpha, gamma, zeta) 分别是 I、距离(惩罚)、健康邻居 的权重
    """
    if not hasattr(uav, "B") or not uav.B:
        # 没有边界点就原地
        if not hasattr(uav, "bp"):
            uav.BP = []
        return 8  # 停

    M = burning_state
    Hh, Ww = M.shape  # 注意：我们用 M[x,y]
    ux, uy = int(uav.x), int(uav.y)
    a, b, c = weights
    max_dist = float(max(Hh, Ww))

    # 8邻域方向
    nbrs = [(-1,-1), (0,-1), (1,-1),
            (-1, 0),          (1, 0),
            (-1, 1), (0, 1),  (1, 1)]

    scores = []
      # === 其他无人机已占用的目标（避免冲突）===
    occupied_targets = set()
    if hasattr(env, "uavs"):
        for v in env.uavs:
            if v is uav:
                continue
            tgt = getattr(v, "current_target", None)
            if tgt is not None:
                # 保守起见转成整数网格坐标
                occupied_targets.add((int(tgt[0]), int(tgt[1])))

    for (x, y) in uav.B:
        if not (0 <= x < Ww and 0 <= y < Hh):
            scores.append(-1e9)
            continue

        v = M[x, y]
        # I：火强（0<v<=1 或 ==1 视为在燃；其余记 0）
        I = float(np.clip(v if ((0 < v <= 1.0) or np.isclose(v, 1.0)) else 0.0, 0.0, 1.0))

        # D：距离（0..1），越近越好 → 分数里是 -γ*D
        D = float(np.hypot(ux - x, uy - y) / (max_dist + 1e-6))

        # H：8邻域健康格子比例（健康=未着火=0）
        healthy = 0
        total = 0
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < Ww and 0 <= ny < Hh:
                total += 1
                healthy += int(M[nx, ny] == 0)
        H = (healthy / total) if total > 0 else 0.0

        score = a * I - b * D + c * H

        if (x, y) in occupied_targets:
            # 方案A：硬禁用（绝不选中）
            score = -1e9

        scores.append(float(score))

    # 保存与 uav.B 对齐的优先级列表
    uav.BP = scores

    # 选择最高优先级点
    best_idx = int(np.argmax(scores))
    best_pt = uav.B[best_idx]
    uav.current_target = best_pt

    # 生成一步动作（8方向+原地；索引与你现有一致）
    action_set = [
        np.array([1,0]),  np.array([1,1]),  np.array([0,1]),
        np.array([-1,1]), np.array([-1,0]), np.array([-1,-1]),
        np.array([0,-1]), np.array([1,-1]), np.array([0,0])
    ]

    tx, ty = best_pt
    dx, dy = tx - ux, ty - uy
    # if dx == 0 and dy == 0:
    #     env.grid['burning_state'][dx, dy] == 3.0
    #     return 8  # 已在目标点，原地（或你在别处触发投弹）

    # 归一化到最近的离散方向
    vec = np.array([np.sign(dx), np.sign(dy)], dtype=float)
    # 从 action_set 里选最接近 vec 的方向
    best_action = min(range(len(action_set)),
                      key=lambda i: np.linalg.norm(
                          (action_set[i] / (np.linalg.norm(action_set[i]) + 1e-8)) - 
                          (vec / (np.linalg.norm(vec) + 1e-8))
                      ))
    return best_action
import numpy as np

def step_attack_flank_minimal(uav, burning_state, env, weights=(0, 1, 0)):
    """
    侧翼：与 head 版完全同风格。
    - 依赖 uav.B_flank: list[(x,y)]
    - burning_state: env.grid['burning_state']，索引 [x, y]
    - weights: (alpha, gamma, zeta) 分别是 I、距离(惩罚)、健康邻居 的权重
    产出：
    - uav.BP_flank: 与 uav.B_flank 对齐的优先级
    - uav.current_target_flank: 选中的目标点
    - 返回：一步动作索引（0..8）
    """
    if not hasattr(uav, "B_flank") or not uav.B_flank:
        if not hasattr(uav, "BP_flank"):
            uav.BP_flank = []
        return 8  # 停

    M = burning_state
    Hh, Ww = M.shape  # 注意：我们用 M[x,y]
    ux, uy = int(uav.x), int(uav.y)
    a, b, c = weights
    max_dist = float(max(Hh, Ww))

    # 8邻域方向
    nbrs = [(-1,-1), (0,-1), (1,-1),
            (-1, 0),          (1, 0),
            (-1, 1), (0, 1),  (1, 1)]

    scores = []
    occupied_targets = set()
    if hasattr(env, "uavs"):
        for v in env.uavs:
            if v is uav:
                continue
            tgt = getattr(v, "current_target", None)
            if tgt is not None:
                # 保守起见转成整数网格坐标
                occupied_targets.add((int(tgt[0]), int(tgt[1])))
    for (x, y) in uav.B_flank:
        if not (0 <= x < Ww and 0 <= y < Hh):
            scores.append(-1e9)
            continue

        v = M[x, y]
        # I：火强（限制到 0..1）
        I = float(np.clip(v if ((0 < v <= 1.0) or np.isclose(v, 1.0)) else 0.0, 0.0, 1.0))

        # D：距离（0..1），越近越好 → 分数里是 -γ*D
        D = float(np.hypot(ux - x, uy - y) / (max_dist + 1e-6))

        # H：8邻域健康格子比例
        healthy, total = 0, 0
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < Ww and 0 <= ny < Hh:
                total += 1
                healthy += int(M[nx, ny] == 0)
        H = (healthy / total) if total > 0 else 0.0

        score = a * I - b * D + c * H
        if (x, y) in occupied_targets:
            # 方案A：硬禁用（绝不选中）
            score = -1e9
        scores.append(float(score))

    uav.BP_flank = scores
    best_idx = int(np.argmax(scores))
    best_pt = uav.B_flank[best_idx]
    uav.current_target = best_pt

    # 生成一步动作（与你现有索引一致）
    action_set = [
        np.array([1,0]),  np.array([1,1]),  np.array([0,1]),
        np.array([-1,1]), np.array([-1,0]), np.array([-1,-1]),
        np.array([0,-1]), np.array([1,-1]), np.array([0,0])
    ]

    tx, ty = best_pt
    dx, dy = tx - ux, ty - uy
    vec = np.array([np.sign(dx), np.sign(dy)], dtype=float)

    best_action = min(
        range(len(action_set)),
        key=lambda i: np.linalg.norm(
            (action_set[i] / (np.linalg.norm(action_set[i]) + 1e-8)) -
            (vec / (np.linalg.norm(vec) + 1e-8))
        )
    )
    return best_action


def step_attack_tail_minimal(uav, burning_state, env, weights=(0, 1, 0)):
    """
    尾部：与 head/flank 版同风格。
    - 依赖 uav.B_tail: list[(x,y)]
    - burning_state: env.grid['burning_state']，索引 [x, y]
    - weights: (alpha, gamma, zeta) 分别是 I、距离(惩罚)、健康邻居 的权重
    产出：
    - uav.BP_tail: 与 uav.B_tail 对齐的优先级
    - uav.current_target_tail: 选中的目标点
    - 返回：一步动作索引（0..8）
    """
    if not hasattr(uav, "B_tail") or not uav.B_tail:
        if not hasattr(uav, "BP_tail"):
            uav.BP_tail = []
        return 8  # 停

    M = burning_state
    Hh, Ww = M.shape  # 注意：我们用 M[x,y]
    ux, uy = int(uav.x), int(uav.y)
    a, b, c = weights
    max_dist = float(max(Hh, Ww))

    nbrs = [(-1,-1), (0,-1), (1,-1),
            (-1, 0),          (1, 0),
            (-1, 1), (0, 1),  (1, 1)]

    scores = []
    occupied_targets = set()
    if hasattr(env, "uavs"):
        for v in env.uavs:
            if v is uav:
                continue
            tgt = getattr(v, "current_target", None)
            if tgt is not None:
                # 保守起见转成整数网格坐标
                occupied_targets.add((int(tgt[0]), int(tgt[1])))
    for (x, y) in uav.B_tail:
        if not (0 <= x < Ww and 0 <= y < Hh):
            scores.append(-1e9)
            continue

        v = M[x, y]
        I = float(np.clip(v if ((0 < v <= 1.0) or np.isclose(v, 1.0)) else 0.0, 0.0, 1.0))

        D = float(np.hypot(ux - x, uy - y) / (max_dist + 1e-6))

        healthy, total = 0, 0
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < Ww and 0 <= ny < Hh:
                total += 1
                healthy += int(M[nx, ny] == 0)
        H = (healthy / total) if total > 0 else 0.0

        score = a * I - b * D + c * H
        if (x, y) in occupied_targets:
            # 方案A：硬禁用（绝不选中）
            score = -1e9
        scores.append(float(score))

    uav.BP_tail = scores
    best_idx = int(np.argmax(scores))
    best_pt = uav.B_tail[best_idx]
    uav.current_target = best_pt

    action_set = [
        np.array([1,0]),  np.array([1,1]),  np.array([0,1]),
        np.array([-1,1]), np.array([-1,0]), np.array([-1,-1]),
        np.array([0,-1]), np.array([1,-1]), np.array([0,0])
    ]

    tx, ty = best_pt
    dx, dy = tx - ux, ty - uy
    vec = np.array([np.sign(dx), np.sign(dy)], dtype=float)

    best_action = min(
        range(len(action_set)),
        key=lambda i: np.linalg.norm(
            (action_set[i] / (np.linalg.norm(action_set[i]) + 1e-8)) -
            (vec / (np.linalg.norm(vec) + 1e-8))
        )
    )
    return best_action


def compute_boundary_explore_rate(env, map_info):
    """
    真·边界：env.grid['burning_state'] 中，火格且八邻域至少一个为0（健康）
    已知边界：在这些边界火格上，map_info 已识别为火/烧尽(>0 或≈2)，
              且存在至少一个邻居在 map_info 中被识别为健康(=0)
    返回 (rate, true_cnt, known_cnt)
    """
    M = env.grid["burning_state"]
    H, W = M.shape

    # 火格（含烧尽）
    is_fire = ((M > 0) & (M <= 1)) | np.isclose(M, 2.0)

    dirs8 = [(-1,-1),(-1,0),(-1,1),
             (0,-1),        (0,1),
             (1,-1), (1,0), (1,1)]

    # —— 真·边界：火格且至少1个邻居为健康(=0)
    has_healthy_neighbor_true = np.zeros_like(is_fire, dtype=bool)

    for dx, dy in dirs8:
        nx = np.clip(np.arange(H)[:, None] + dx, 0, H-1)
        ny = np.clip(np.arange(W)[None, :] + dy, 0, W-1)
        neigh_true = M[nx, ny]
        has_healthy_neighbor_true |= (neigh_true == 0)

    boundary_true = is_fire & has_healthy_neighbor_true
    true_cnt = int(boundary_true.sum())
    if true_cnt == 0:
        return 1.0, 0, 0  # 无边界视为已掌握

    # —— 已知条件（基于 map_info）——
    # 1) 边界火格在 map_info 中被识别为火/烧尽（不是仅非未知）
    known_fire = (map_info > 0) | np.isclose(map_info, 2.0)

    # 2) 至少一个邻居在 map_info 中被识别为健康(=0)
    has_known_healthy_neighbor = np.zeros_like(boundary_true, dtype=bool)
    for dx, dy in dirs8:
        nx = np.clip(np.arange(H)[:, None] + dx, 0, H-1)
        ny = np.clip(np.arange(W)[None, :] + dy, 0, W-1)
        neigh_map = map_info[nx, ny]
        has_known_healthy_neighbor |= (neigh_map == 0)

    # known_boundary_mask = boundary_true & known_fire & has_known_healthy_neighbor
    known_boundary_mask = known_fire & has_known_healthy_neighbor
    known_cnt = int(known_boundary_mask.sum())
    rate = known_cnt / true_cnt

    return float(rate), true_cnt, known_cnt

import numpy as np

import numpy as np

def heuristic_action_using_map(uav, obs):
    """
    最简版探索启发式：
    - memory==0：朝地图中心靠近
    - memory==1：把所有“边界火点”周围的 -1 未知格收集到 uav.unknown，选最近未知格靠近；
                 若没有未知格则返回 8（原地）
    说明：
    - 边界火点：burning_state 里 0<v<=1 的格子
    - 方向索引与环境一致：0..7 为 8 邻域，8 为原地
    """
    M = uav.map_info                     # 你代码里用的是 M[x,y] 索引
    H, W = M.shape
    x0, y0 = int(uav.x), int(uav.y)

    # 8方向 + 停
    dirs = [
        ( 1, 0), ( 1, 1), ( 0, 1), (-1, 1),
        (-1, 0), (-1,-1), ( 0,-1), ( 1,-1), ( 0, 0)
    ]

    memory = float(obs[0])
    # === 第一阶段：朝火场中心 ===
    if memory == 0.0:
        cx, cy = 15, 25        # 地图中心（不再用固定 25,25）
        dx = int(np.sign(cx - x0))
        dy = int(np.sign(cy - y0))
        step = (dx, dy)
        # 映射到动作索引
        try:
            return dirs.index(step)
        except ValueError:
            return 8

    # === 第二阶段：基于 map_info 的未知优先 ===
    # 收集“边界火点附近的 -1 未知格”
    unknown_set = set()

    def inb(x, y): return 0 <= x < H and 0 <= y < W
    def is_fire(v): return (0.0 < v <= 1.0)  # 在燃格
    def is_unknown(v): return (v == -1)

    # 遍历所有在燃格（边界火点的近似），把其 8 邻域中的 -1 未知加入候选
    xs, ys = np.where((M > 0.0) & (M <= 1.0))
    for x, y in zip(xs, ys):
        for dx, dy in dirs[:-1]:  # 不含原地
            nx, ny = x + dx, y + dy
            if inb(nx, ny) and is_unknown(M[nx, ny]):
                unknown_set.add((nx, ny))

    # 写回 uav.unknown（去重后的列表）
    uav.unknown = list(unknown_set)

    if not uav.unknown:
        if not hasattr(uav, "v_pref"):
            uav.v_pref = np.array([1.0, 0.0], dtype=float)  # 初始偏好方向

        def is_edge(v): 
            return (0.0 < v <= 1.0) or np.isclose(v, 3.0)

        def angle_cost(u, v):
            u = np.asarray(u, float); v = np.asarray(v, float)
            if np.allclose(u, 0): return 1e3
            u = u / (np.linalg.norm(u) + 1e-8)
            v = v / (np.linalg.norm(v) + 1e-8)
            return np.linalg.norm(u - v)

        def left_of(vec):
            vx, vy = vec
            return np.array([-vy, vx], float)
        def right_of(vec):
            vx, vy = vec
            return np.array([vy, -vx], float)  # 右转 90°
        edge_moves = []
        for idx, (dx, dy) in enumerate(dirs[:-1]):  # 不考虑原地
            nx, ny = x0 + dx, y0 + dy
            if 0 <= nx < H and 0 <= ny < W and is_edge(M[nx, ny]):

                L = left_of((dx, dy))
                lx = x0 + int(np.clip(np.round(L[0]), -1, 1))
                ly = y0 + int(np.clip(np.round(L[1]), -1, 1))
                left_is_edge = (0 <= lx < H and 0 <= ly < W and is_edge(M[lx, ly]))
                score = angle_cost((dx, dy), uav.v_pref) - (0.15 if left_is_edge else 0.0)

                # R = right_of((dx, dy))
                # rx = x0 + int(np.clip(np.round(R[0]), -1, 1))
                # ry = y0 + int(np.clip(np.round(R[1]), -1, 1))
                # right_is_edge = (0 <= rx < H and 0 <= ry < W and is_edge(M[rx, ry]))
                # score = angle_cost((dx, dy), uav.v_pref) - (0.15 if right_is_edge else 0.0)

                edge_moves.append((score, idx))

        if edge_moves:
            edge_moves.sort(key=lambda t: t[0])
            best_idx = edge_moves[0][1]
            uav.v_pref = np.array(dirs[best_idx][:2], float)  # 更新偏好方向
            return best_idx

        # 2) 8 邻域没有边缘 → 找最近边缘点，朝它迈一步（仍不会原地）
        xs, ys = np.where(((M > 0.0) & (M <= 1.0)) | (np.isclose(M, 3.0)))
        if xs.size > 0:
            # 找全局最近边缘点
            d2 = (xs - x0) ** 2 + (ys - y0) ** 2
            k = int(np.argmin(d2))
            tx, ty = int(xs[k]), int(ys[k])
            dx, dy = int(np.sign(tx - x0)), int(np.sign(ty - y0))
            step = (dx, dy)
            if step in dirs:
                uav.v_pref = np.array(step, float)
                return dirs.index(step)

        # 实在没目标
        return 8


    # 选择最近的未知格（欧氏或曼哈顿都行；这里用欧氏）
    def dist2(p):  # 返回距离平方，省一个开方
        return (p[0] - x0) ** 2 + (p[1] - y0) ** 2

    tgt = min(uav.unknown, key=dist2)

    # 朝目标未知格迈一步（离散到 8 邻域）
    dx = int(np.sign(tgt[0] - x0))
    dy = int(np.sign(tgt[1] - y0))
    step = (dx, dy)

    # print(uav.unknown)
    try:
        return dirs.index(step)
    except ValueError:
        return 8

import numpy as np

import numpy as np

def heuristic_action_using_map_fuxian(uav, env=None):
    """
    阶段1（memory=0）：直接朝 (15,25) 靠近
    阶段2（memory=1）：只在边缘集合（B ∪ B_flank ∪ B_tail）上，严格顺时针巡边
      - 只允许顺时针方向（相对上一步方向 +k, k>0）
      - 若没有合法顺时针下一步：原地等待（保持方向约束）
    返回：动作索引 0..8
    """
    # 与工程一致的方向/动作定义（函数内局部，不污染全局）
    dirs8 = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    action_set = [
        np.array([1,0]), np.array([1,1]), np.array([0,1]),
        np.array([-1,1]), np.array([-1,0]), np.array([-1,-1]),
        np.array([0,-1]), np.array([1,-1]), np.array([0,0])
    ]

    def vec_to_idx(dx, dy):
        if dx == 0 and dy == 0: return 8
        v = np.array([dx,dy], float); v /= (np.linalg.norm(v)+1e-8)
        best = min(range(8), key=lambda i: np.linalg.norm(
            v - (action_set[i] / (np.linalg.norm(action_set[i])+1e-8))
        ))
        return best

    def edge_union(uav_):
        S = set()
        S |= set(getattr(uav_, "B", []))
        S |= set(getattr(uav_, "B_flank", []))
        S |= set(getattr(uav_, "B_tail", []))
        return S

    def fire_on_right(M, nx, ny, nd):
        """判断从 nd 方向走到 (nx,ny) 后，其右侧是否为火（0<val<3）"""
        if M is None: return False
        H, W = M.shape
        rd = (nd + 1) % 8                      # 右侧方向
        rx, ry = nx + dirs8[rd][0], ny + dirs8[rd][1]
        if 0 <= rx < H and 0 <= ry < W:
            v = M[rx, ry]
            return (v > 0.0) and (v < 3.0)     # 把“烧尽=2”也算作火侧
        return False

    # -------- 阶段判定 --------
    ux, uy = int(uav.x), int(uav.y)
    memory = bool(getattr(uav, "memory_flag", False))

    # 阶段1：未见到火 -> 朝 (15,25)
    if not memory:
        tx, ty = 15, 25
        return vec_to_idx(int(np.sign(tx-ux)), int(np.sign(ty-uy)))

    # 阶段2：已见到火 -> 只沿边 & 严格顺时针
    M = getattr(uav, "map_info", None)
    edge_set = edge_union(uav)



    # 初始化方向：选一个邻边缘点，且该步“右侧为火”，以保证顺时针语义
    if getattr(uav, "edge_dir", None) is None:
        init = None
        if edge_set:
            for i, (dx, dy) in enumerate(dirs8):
                nx, ny = ux + dx, uy + dy
                if (nx, ny) in edge_set and fire_on_right(M, nx, ny, i):
                    init = i
                    break
            # 若找不到“右侧为火”的邻居，也可退而求其次选任一邻边缘点
            if init is None:
                for i, (dx, dy) in enumerate(dirs8):
                    nx, ny = ux + dx, uy + dy
                    if (nx, ny) in edge_set:
                        init = i
                        break
        uav.edge_dir = 0 if init is None else init

    # 严格顺时针：只允许相对当前方向“正向转动” (+k, k=1..7)，绝不左转
    for k in range(1, 8):  # 依次尝试 +1,+2,...,+7
        nd = (uav.edge_dir + k) % 8
        dx, dy = dirs8[nd]
        nx, ny = ux + dx, uy + dy
        # 下一步必须仍在边集合，且右侧为火（右手贴边）
        if (nx, ny) in edge_set and fire_on_right(M, nx, ny, nd):
            uav.edge_dir = nd
            return vec_to_idx(dx, dy)

    # 若所有顺时针候选都不合法 → 原地等待，保持"严格顺时针"约束
    return 8


# def step_comm_support(uav, env, dist_hover: float = 3.0) -> int:
#     """
#     通信保障任务行为逻辑：
#     - UAV 朝人群质心（rescuer_center）移动
#     - 如果 UAV 已经在质心附近（距离小于通信半径），则原地悬停
#     - 当 UAV 的通信半径覆盖到任意一个救援人员时，视为成功通信，time_since_last_comm = 0
    
#     返回动作索引 0..8
#     """
#     if not hasattr(env, "rescuers") or len(env.rescuers) == 0:
#         return 8  # 没有救援人员，原地
    
#     # 计算人群质心
#     # 更新逻辑：按不同位置去重后求平均，而不是按数量加权
#     # 例如：9个在(0,0)，1个在(10,10)，质点应该是(5,5)而不是(1,1)
#     rescuer_center_x = 0.0
#     rescuer_center_y = 0.0
#     if len(env.rescuers) > 0:
#         # 收集所有不同的位置（四舍五入到整数，然后去重）
#         unique_positions = set()
#         for rescuer in env.rescuers:
#             # 将位置四舍五入到整数，用于去重
#             pos_key = (int(round(rescuer.x)), int(round(rescuer.y)))
#             unique_positions.add(pos_key)
        
#         # 计算所有不同位置的平均值
#         if len(unique_positions) > 0:
#             for px, py in unique_positions:
#                 rescuer_center_x += px
#                 rescuer_center_y += py
#             rescuer_center_x /= len(unique_positions)
#             rescuer_center_y /= len(unique_positions)
    
#     # 注意：信息融合和time_since_last_comm重置现在在update_communication_coverage中统一处理
#     # 这里只需要计算到质心的距离并移动
#     uav_comm_radius = getattr(env, "uav_comm_radius", 10)
#     ux, uy = uav.x, uav.y
    
#     # 计算到人群质心的距离
#     dist_to_center = np.hypot(ux - rescuer_center_x, uy - rescuer_center_y)
    
#     # 如果已经在质心附近（距离小于通信半径），则原地悬停
#     if dist_to_center < 1:
#         return 8  # 原地悬停
    
#     # 朝人群质心移动
#     dx = int(np.sign(rescuer_center_x - ux))
#     dy = int(np.sign(rescuer_center_y - uy))
#     move = (dx, dy)
    
#     action_set = [
#         np.array([1,0]),  np.array([1,1]),  np.array([0,1]),
#         np.array([-1,1]), np.array([-1,0]), np.array([-1,-1]),
#         np.array([0,-1]), np.array([1,-1]), np.array([0,0])
#     ]
    
#     for idx, a in enumerate(action_set):
#         if (a[0] == move[0]) and (a[1] == move[1]):
#             return idx
    
#     return 8

import numpy as np

def step_comm_support(uav, env, aoi_threshold: int = 5) -> int:
    """
    通信保障任务（协作版）：
    - 所有选了“通信保障”任务的 UAV 视为一个小系统；
    - 每个时间步为所有通信 UAV 规划“站位点”，使总覆盖（人数 + AoI 权重）尽量大；
    - 分配顺序：先给第一个 UAV 选全局最优，再给第二个选时对重叠覆盖的人加惩罚，
      从而鼓励多个 UAV 分工覆盖不同人群，而不是挤在一堆。
    - 本函数在被某个 UAV 调用时，会基于当前 env 状态做一次完整规划，
      然后返回当前这个 UAV 对应站位的移动动作（或悬停）。
    """

    # 1）环境中没有人 / 没有 UAV：直接原地
    if not hasattr(env, "rescuers") or len(env.rescuers) == 0:
        return 8
    if not hasattr(env, "uavs") or len(env.uavs) == 0:
        return 8

    # 找出当前这个 UAV 在 env.uavs 中的索引
    try:
        my_idx = env.uavs.index(uav)
    except ValueError:
        # 不在列表里，防御性处理
        my_idx = None

    # 通信任务对应的高层 option，按你当前代码是 1
    COMM_OPTION = 1
    comm_indices = [
        idx for idx, u in enumerate(env.uavs)
        if getattr(u, "option", 0) == COMM_OPTION
    ]
    # 如果没有 UAV 在执行通信任务，就原地
    if not comm_indices:
        return 8

    # 如果当前 UAV 不在通信 UAV 之中，也直接原地
    if (my_idx is None) or (my_idx not in comm_indices):
        return 8

    # -----------------------------
    # 2）构造候选站位点：用“所有救援人员的位置（去重后）”
    # -----------------------------
    uav_comm_radius = float(getattr(env, "uav_comm_radius", 10.0))
    R = uav_comm_radius

    # unique rescuer positions（四舍五入到整数用于聚类）
    unique_positions = {}
    for i, rescuer in enumerate(env.rescuers):
        key = (int(round(float(rescuer.x))), int(round(float(rescuer.y))))
        # 一个位置可能对应多个 rescuer，这里只存一个代表点的连续坐标
        if key not in unique_positions:
            unique_positions[key] = (float(rescuer.x), float(rescuer.y))

    if not unique_positions:
        return 8

    candidate_centers = list(unique_positions.values())

    # 预先算出：每个候选中心能覆盖到哪些 rescuer（下标）
    rescuer_positions = np.array([[float(r.x), float(r.y)] for r in env.rescuers])
    candidates_cover = []  # List[List[int]]
    for cx, cy in candidate_centers:
        dx = rescuer_positions[:, 0] - cx
        dy = rescuer_positions[:, 1] - cy
        dist_sq = dx * dx + dy * dy
        covered_idx = np.where(dist_sq <= (R * R))[0].tolist()
        candidates_cover.append(covered_idx)

    # AoI 数组，避免反复访问属性
    aoi_array = np.array([getattr(r, "AoI", 0) for r in env.rescuers], dtype=float)
    # 简单归一化一下 AoI（避免过大）
    max_aoi = max(aoi_array.max(), 1.0)
    norm_aoi = aoi_array / max_aoi

    # -----------------------------
    # 3）为所有通信 UAV 做一次 greedy 规划
    #     - 顺序固定为 comm_indices 升序，避免调用顺序造成不一致
    # -----------------------------
    # 超参：你可以后面自己调
    w_base = 1.0          # 覆盖一个人的基础收益
    w_aoi = 1.0           # AoI 权重（越大越偏向救落单的人）
    gamma_overlap = 0.7   # 重叠覆盖惩罚 [0,1]，越大越不喜欢和别人覆盖同一个人
    lambda_move = 0.1     # 走远路的惩罚

    # 将通信 UAV 按索引排序
    comm_indices_sorted = sorted(comm_indices)

    # 记录已分配给前若干 UAV 的“已经覆盖过的人”的集合（用于重叠惩罚）
    covered_sets = []  # List[Set[int]]
    # 规划结果：uav_idx -> (tx, ty)
    assigned_targets = {}

    for u_idx in comm_indices_sorted:
        uav_k = env.uavs[u_idx]
        ux, uy = float(uav_k.x), float(uav_k.y)

        best_score = -1e18
        best_center = None
        best_covered = set()

        # 遍历所有候选中心
        for c_idx, (cx, cy) in enumerate(candidate_centers):
            covered = candidates_cover[c_idx]
            if not covered:
                continue

            # 计算该站位点的“覆盖收益”
            # base_weight_i = w_base + w_aoi * norm_aoi[i]
            # 如果已经被前面的 UAV 覆盖过，则乘以 (1 - gamma_overlap)
            base_weights = w_base + w_aoi * norm_aoi[covered]
            if covered_sets and gamma_overlap > 1e-6:
                overlap_mask = np.zeros_like(base_weights)
                # 标记哪些 rescuer 已经被之前的 UAV 覆盖过
                for S in covered_sets:
                    # S 是 set[int]，这里求交集映射一下
                    # 把 covered 中在 S 里的 index 设为 1
                    for j, res_idx in enumerate(covered):
                        if res_idx in S:
                            overlap_mask[j] = 1.0
                # 应用重叠惩罚
                base_weights = base_weights * (1.0 - gamma_overlap * overlap_mask)

            coverage_score = float(base_weights.sum())

            # 距离惩罚（离得越近越好）
            dist = abs(cx - ux) + abs(cy - uy)  # 曼哈顿距离即可
            move_cost = lambda_move * dist

            total_score = coverage_score - move_cost

            if total_score > best_score:
                best_score = total_score
                best_center = (cx, cy)
                best_covered = set(covered)

        if best_center is None:
            # 没有任何候选点有收益，就让它原地
            assigned_targets[u_idx] = (ux, uy)
            covered_sets.append(set())
        else:
            assigned_targets[u_idx] = best_center
            covered_sets.append(best_covered)

    # -----------------------------
    # 4）拿到当前这个 UAV 的站位目标，生成低层动作
    # -----------------------------
    tx, ty = assigned_targets.get(my_idx, (float(uav.x), float(uav.y)))
    ux, uy = float(uav.x), float(uav.y)

    # 如果已经非常接近目标点，就原地（避免抖动）
    dist_to_target = np.hypot(ux - tx, uy - ty)
    if dist_to_target < 0.5:
        return 8  # 悬停

    # 朝站位点移动（8 邻域）
    dx = int(np.sign(tx - ux))
    dy = int(np.sign(ty - uy))
    move = (dx, dy)

    action_set = [
        np.array([1,0]),  np.array([1,1]),  np.array([0,1]),
        np.array([-1,1]), np.array([-1,0]), np.array([-1,-1]),
        np.array([0,-1]), np.array([1,-1]), np.array([0,0])
    ]

    for idx, a in enumerate(action_set):
        if (a[0] == move[0]) and (a[1] == move[1]):
            return idx

    # 理论上不会走到这里，兜底悬停
    return 8

def explore(env, k=3):
    print(f"🖥 当前使用设备: {device}")
    obs_list, _ = env.reset()
    done= False
    t=0
    for t in range(300):
        if t<150:
            actions = []
            # 遍历每架无人机
            for obs in obs_list:
                action = heuristic_action(obs)  
                actions.append(action)
            for uav in env.uavs:
                # b = select_band_and_boundary(uav)
                q1,q2,q3 = compute_boundary_explore_rate(env,uav.map_info)
                # metrics = compute_explore_metrics(uav.map_info)
                # global_cov = metrics["global_coverage"]
                # boundary_comp = metrics["boundary_completeness"]
                # num_bianjie = metrics["num_boundary_cells"]
                print(f"探索率:{q1},真实格子{q2},已知格子{q3}")
        elif t>150 :
            actions = []
            # 遍历每架无人机
            for uav in env.uavs:
                b = select_band_and_boundary(uav)
                # print(f"{uav.uav_id}的边界点是{uav.B}")
                action = step_attack_head_minimal(uav,uav.map_info,env)
                print(uav.current_target)
                print(uav.B)
                # print(f"{uav.uav_id}的边界点优先级为{uav.BP}")
                if uav.B==[]:
                    for obs in obs_list:
                        action = heuristic_action(obs)
                actions.append(action)

        t+=1

        # 将 joint actions 传给环境
        next_obs_list, rewards, terminated, truncated, infos = env.step(actions)

        env.render()
        obs_list = next_obs_list


   

if __name__ == "__main__":
    # 创建环境
    env = GridEnv()
    explore(env)
