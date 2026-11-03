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

def heuristic_action(obs):
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

    if memory == 0.0:
        # 第一阶段: 朝火中心
        fire_center = np.array([25,25])
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
