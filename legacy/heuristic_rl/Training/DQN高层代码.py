# -*- coding: utf-8 -*-
from collections import deque
from matplotlib import pyplot as plt
import numpy as np
from typing import List, Dict, Any
import sys, os

import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Env.低层环境 import GridEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from 探索任务 import (                                     # 启发式工具
    heuristic_action,               # 侦察：obs_low -> action_idx
    heuristic_action_using_map,
    select_band_and_boundary,       # 预处理，写入 uav.B
    select_flank_by_ybands,
    select_tail_by_xband,
    step_attack_head_minimal,        # 火头灭火：uav/map/env -> action_idx
    step_attack_flank_minimal,
    step_attack_tail_minimal,
    step_go_charge,
    heuristic_action_using_map_fuxian,
    step_communicate_nearest,       # 通信子任务：靠近最近无人机，距离<阈值则停下
    step_comm_support,              # 通信保障任务：朝向未被覆盖的救援人员
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode="valid")

class DuelingQNet(nn.Module):
    def __init__(self, in_dim, n_options, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        # 价值流 V(s)
        self.V = nn.Linear(hidden, 1)
        # 优势流 A(s,a)
        self.A = nn.Linear(hidden, n_options)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        V = self.V(x)                         # [B,1]
        A = self.A(x)                         # [B,n_options]
        # dueling 合成：Q = V + (A - mean(A))
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q
# ============== 1) 高层策略 ==============
class HighLevelPolicy:
    def __init__(self, n_options=5, obs_dim=14, gamma=0.99, lr=1e-3,
                 eps_start=1.0, eps_end=0.05, eps_decay=500,
                 buf_size=1000000, batch_size=128, target_tau=0.005, device=device,
                 use_topology_obs=True, use_topology_exploration=True):
        """
        参数:
            use_topology_obs: 是否使用拓扑观测（火头/侧翼/火尾长度）
            use_topology_exploration: 是否使用拓扑感知的探索调度（仅在有拓扑观测时有效）
        """
        # obs_dim=14: g(7) + battery_level(1) + coverage_rate(1) + 
        #             self_dist_to_rescuer_center(1) + min_team_dist_to_rescuer_center(1) + task_ratio(3)
        # g[0-6]: wind_speed, rate, fire_diff, head_len, flank_len, tail_len, coverage_rate
        # 已删除: max_AoI, time_since_last_comm
        # 新增: task_ratio (3维: 探索、通信、补给占比)

        self.n_options = n_options
        self.gamma = gamma
        self.device = device
        self.n_step = 5                        # ← 你的目标里用了 γ**5
        self.use_action_mask = True
        
        # 拓扑模式配置（用于消融实验）
        self.use_topology_obs = use_topology_obs
        self.use_topology_exploration = use_topology_exploration and use_topology_obs  # 无拓扑时强制关闭

        self.online = DuelingQNet(obs_dim, n_options).to(device)
        self.target = DuelingQNet(obs_dim, n_options).to(device)
        self.target.load_state_dict(self.online.state_dict())

        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.batch_size = batch_size
        self.target_tau = target_tau
        self.replay = deque(maxlen=buf_size)
        self.eps_s = eps_start; self.eps_e = eps_end; self.eps_d = eps_decay


    def epsilon(self, ep):
        # ep = 当前的 episode index
        frac = min(ep / self.eps_d, 1.0)   # 0 → 1
        base_epsilon = self.eps_s + frac * (self.eps_e - self.eps_s)
        return base_epsilon
    
    def compute_exploration_factor(self, obs_high, episode):
       
        if not self.use_topology_exploration:
            # 无拓扑模式：返回1.0（不调整，退化为标准ε衰减）
            return 1.0
        
        try:
            # 提取观测信息
            explore_rate = float(obs_high[1]) if len(obs_high) > 1 else 0.0  # g[1]: 全局探索率
            head_len = float(obs_high[3]) if len(obs_high) > 3 else 0.0       # g[3]: 火头长度
            flank_len = float(obs_high[4]) if len(obs_high) > 4 else 0.0       # g[4]: 侧翼长度
            tail_len = float(obs_high[5]) if len(obs_high) > 5 else 0.0       # g[5]: 火尾长度
            
            # 确保值在合理范围内
            explore_rate = np.clip(explore_rate, 0.0, 1.0)
            head_len = max(0.0, head_len)
            flank_len = max(0.0, flank_len)
            tail_len = max(0.0, tail_len)
            
            # ========== 触发条件检查 ==========
            # 训练早期阈值
            E_warmup = 80
            is_early_training = episode < E_warmup
            
            # 全局探索率阈值
            explore_rate_threshold = 0.8
            is_high_explore_rate = explore_rate > explore_rate_threshold
            
            
            # 火头长度阈值（明显较小或尚未被发现）
            head_threshold = 1.0 / 10.0  # 原始长度1.0，归一化后为0.1
            is_small_head = head_len <= head_threshold
            
            # 检查是否满足所有触发条件
            all_conditions_met = (
                is_early_training and
                is_high_explore_rate and
                is_small_head
            )
            
            # ========== 调度逻辑 ==========
            if all_conditions_met:
                # 触发额外随机性：增加ε，防止策略过早稳定
                # 使用较大的因子（2.0-3.0），在训练早期提供足够的探索扰动
                # 因子随episode逐渐衰减，避免长期干扰
                max_factor = 2.5  # 最大随机因子
                min_factor = 1.5  # 最小随机因子（仍高于标准）
                
                # 在warmup期间线性衰减
                progress = episode / float(E_warmup)  # [0, 1]
                exploration_factor = max_factor - (max_factor - min_factor) * progress
                exploration_factor = np.clip(exploration_factor, min_factor, max_factor)
                
                return float(exploration_factor)
            else:
                return 1.0
            
        except Exception as e:
            # 出错时返回1.0（不调整，保持标准行为）
            return 1.0
    
    import numpy as np

    

    def select_option(self, obs_high, ep, env=None, uav_idx=None, is_eval=False):
        """
        纯强化学习任务选择函数（已移除启发式分支）
        
        决策逻辑：
        1. 电量约束 mask（保留）
        2. epsilon-greedy 决策：
           - 随机：从合法 option 中采样
           - 否则：argmax Q(s, option)
        
        启发式仅用于探索强度调节（通过compute_exploration_factor调整epsilon）
        
        参数:
            is_eval: 是否为评估模式（评估时epsilon=0，完全关闭探索）
        """
        n_opts = int(self.n_options)
        
        # 读取电量（battery_level在索引8）
        elec = float(obs_high[7])  # battery_level
        
        # === 构造动作掩码：电量掩码 ===
        mask = np.ones(n_opts, dtype=bool)  # 默认所有动作都允许
        
        # 当电量小于0.05时，只能选择补给（option=2）
        if elec < 0.05:
            mask.fill(False)
            if n_opts >= 3:  # 确保有补给选项
                mask[2] = True  # 只允许补给
            else:
                mask[0] = True  # 如果没有补给选项，至少允许探索

        # Q 估计
        q = self.online(torch.from_numpy(obs_high).float().unsqueeze(0).to(self.device))
        q = q.detach().cpu().numpy()[0]

        # === 纯RL决策：epsilon-greedy ===
        if is_eval:
            # 评估模式：完全关闭探索，使用greedy policy
            epsilon = 0.0
        else:
            # 训练模式：计算基础epsilon
            base_epsilon = self.epsilon(ep)
            # 使用探索强度调度模块调整epsilon（启发式仅影响探索概率）
            # 传入episode参数，用于判断是否在训练早期
            exploration_factor = self.compute_exploration_factor(obs_high, ep)
            epsilon = base_epsilon * exploration_factor
            epsilon = np.clip(epsilon, 0.0, 1.0)  # 限制在[0, 1]
        
        u = np.random.rand()
        
        if u < epsilon:
            # 随机探索：只从"有效动作集合"里抽
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                # 极端情况：至少允许一个动作
                return 0
            return int(np.random.choice(valid))
        else:
            # 贪心选择：应用掩码后选择Q值最大的动作
            q[~mask] = -1e9
            return int(np.argmax(q))





    def push(self, tr):  # tr: dict with obs_high, option, Rk, next_obs_high, done
        self.replay.append(tr)

    def soft_update(self):
        # target ← τ*online + (1-τ)*target
        for tp, p in zip(self.target.parameters(), self.online.parameters()):
            tp.data.mul_(1-self.target_tau).add_(self.target_tau*p.data)


    def update(self):
        if len(self.replay) < self.batch_size: return None
        import random, numpy as np, torch, torch.nn.functional as F

        batch = random.sample(self.replay, self.batch_size)
        obs  = torch.tensor(np.stack([b["obs_high"] for b in batch]), dtype=torch.float32, device=self.device)
        act  = torch.tensor([b["option"] for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        Rk   = torch.tensor([b["Rk"] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        nxt  = torch.tensor(np.stack([b["next_obs_high"] for b in batch]), dtype=torch.float32, device=self.device)
        done = torch.tensor([b["done"] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            # Double DQN：online 选动作
            q_next_online = self.online(nxt)  # [B, n_options]

            if self.use_action_mask:
                # === 掩码：电量小于0.05时只能选择补给 ===
                B, n_opts = q_next_online.shape
                mask_next = torch.ones_like(q_next_online, dtype=torch.bool)  # 默认所有动作都允许

                # 读取电量（battery_level在索引8）
                elec = nxt[:, 7]  # battery_level
                
                # 当电量小于0.05时，只能选择补给（option=2）
                low_battery = elec < 0.05
                if n_opts >= 3:  # 确保有补给选项
                    # 电量小于0.05的样本：只允许补给
                    mask_next[low_battery, :] = False
                    mask_next[low_battery, 2] = True  # 只允许补给
                else:
                    # 如果没有补给选项，至少允许探索
                    mask_next[low_battery, :] = False
                    mask_next[low_battery, 0] = True

                # 应用掩码
                q_next_online = q_next_online.masked_fill(~mask_next, -1e9)

            nxt_act = q_next_online.argmax(dim=1, keepdim=True)  # [B,1]

            # target 评估该动作
            q_next_target = self.target(nxt).gather(1, nxt_act)  # [B,1]
            y = Rk + (1.0 - done) * (self.gamma ** self.n_step) * q_next_target

        q = self.online(obs).gather(1, act)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.opt.step()
        self.soft_update()
        return float(loss.item())

    def _build_mask_from_obs_np(self, obs_high):
        """
        构建动作掩码：当电量小于0.05时，只能选择补给（option=2）
        """
        n_opts = int(self.n_options)
        elec = float(obs_high[7])  # battery_level
        
        # 默认所有动作都允许
        mask = np.ones(n_opts, dtype=bool)
        
        # 当电量小于0.05时，只能选择补给（option=2）
        if elec < 0.05:
            mask.fill(False)
            if n_opts >= 3:  # 确保有补给选项
                mask[2] = True  # 只允许补给
            else:
                mask[0] = True  # 如果没有补给选项，至少允许探索
        
        return mask


    def select_greedy_masked(self, obs_high):
        """评估/测试用：带掩码的贪心选择（不包含随机/启发式）"""
        q = self.online(torch.from_numpy(obs_high).float().unsqueeze(0).to(self.device))
        q = q.detach().cpu().numpy()[0]
        mask = self._build_mask_from_obs_np(obs_high)
        # 如果极端情况下四个都是 False（基本不会发生），至少允许探索
        if not mask.any():
            mask[0] = True
        q[~mask] = -1e9
        return int(np.argmax(q))
# ============== 2) 计算全局宏观特征（可按实验计划逐步丰富） ==============
def estimate_boundary_len(M: np.ndarray) -> float:
    """估算“火边界长度”：统计燃烧/烧尽格子四邻域含健康格的数量并归一化。"""
    H, W = M.shape
    is_fire = ((M > 0) & (M <= 1)) | np.isclose(M, 2.0)
    boundary = 0
    for x in range(H):
        for y in range(W):
            if not is_fire[x, y]:
                continue
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < H and 0 <= ny < W and M[nx, ny] == 0:
                    boundary += 1
                    break
    # 归一化：除以格子总数，得到约 0~1
    return boundary / float(H * W + 1e-6)



def compute_global_features(env: GridEnv, idx: int, use_topology_obs: bool = True) -> np.ndarray:
    """
    抽取宏观状态：风、探索率、火势增长，以及"当前无人机自己观测到的"火头/侧翼/尾部长度。
    在部分通信下，三类长度都基于各自 uav 的局部火势地图（而不是统一用第0架）。
    
    参数:
        use_topology_obs: 是否使用拓扑观测（用于消融实验）
                         如果False，将火头/侧翼/火尾长度置零
    """
    M = env.grid["burning_state"]
    H, W = M.shape
    before_fire = np.count_nonzero(env.grid["burning_state"] > 0)
    #1.风速
    wind_speed = float(env.wind_speed) / 10

    #2.探索率
    rate = env.known_boundary / env.true_boundary

    # print(f"探索率为{rate},已知的边界为{known_cnt},真实的边界为{true_cnt}")

    #3.火势增长
    fire_diff = env.fire_diff / 10

    #2.各火区长度（使用当前 idx 无人机自己的集合）
    if use_topology_obs and 0 <= idx < len(env.uavs):
        uav_i = env.uavs[idx]
        head_len = len(getattr(uav_i, "B", [])) / 10
        flank_len = len(getattr(uav_i, "B_flank", [])) / 10
        tail_len = len(getattr(uav_i, "B_tail", [])) / 10
    else:
        # 无拓扑模式：置零（消融实验）
        head_len = flank_len = tail_len = 0.0

    #4.通信特征（全局共享）
    coverage_rate = float(getattr(env, "coverage_rate", 0.0))
    # 已删除 max_AoI

    return np.array([wind_speed, float(rate), fire_diff, head_len, flank_len, tail_len, 
                     coverage_rate], dtype=np.float32)
import numpy as np

def build_team_aggregate(self, self_idx: int, prev_options, 
                         ammo_tau: float = 0.3, 
                         target_staffing=(1.5, 1, 0.5)) -> np.ndarray:
    """
    聚合队友信息（拼接到高层obs末尾）——10维：
      [4维队友任务占比@t-1（不含自己，顺序无关）] +
      [3维接力需求 need_head/flank/tail] +
      [3维队友到三类目标的最短距离（归一化到0~1）]

    参数
    ----
    target_staffing : (α_head, α_flank, α_tail)，期望同时在线的有效灭火位数量

    要求
    ----
    - 每个 UAV 应有属性：
        .x, .y : 当前整数网格坐标
        .d     : 当前弹药数（与 ammo_max 同量纲）
        .B / .B_flank / .B_tail : 每类目标点集合（list/iterable of (x,y)）
    - 环境/类应可提供：
        self.uavs                      : UAV 列表
        self.grid["burning_state"].shape -> (H, W)
        self.ammo_max  (可选) 或假定为10
    """
    num_uav = len(self.uavs)

    # ---------- 1) 队友任务占比（不含自己） ----------
    task = prev_options.copy()

    task[self.uavs[self_idx].option] -=1
    others_idx = [i for i in range(num_uav) if i != self_idx]
    K = len(others_idx)
    # print(f"我是{self_idx}号无人机，上一步的总体选择为{prev_options},我选的是{self.uavs[self_idx].option},去掉我后的选择为{task}")
    task_ratio =task/ 3  # 归一化为占比
    # print(f"归一化后{task_ratio}")

    # ---------- 2) 接力需求 need_*（基于 t-1 任务 + 弹药阈值） ----------
    # 弹药归一化：与低层保持一致；优先从 env 取 ammo_max，默认10
    ammo_max = float(getattr(self, "ammo_max", 10.0))
    def _eff_count(indices):
        if not indices:
            return 0.0
        cnt = 0
        for i in indices:
            # 允许 UAV 上没有 d 属性时回退为0
            ai = float(getattr(self.uavs[i], "d", 0.0)) / (ammo_max + 1e-6)
            if ai >= ammo_tau:
                cnt += 1
        return float(cnt)
    H, W = self.grid["burning_state"].shape
    diag = np.hypot(10, 10) + 1e-6
    def _collect_points_for_self(attr_name):
        """
        在“部分通信”设定下，火头/侧翼/尾部集合应来源于本机自己的地图。
        这里只收集 self_idx 这架无人机自身维护的集合；
        其他无人机的位置、电量等仍按全局可见处理。
        """
        uav_self = self.uavs[self_idx]
        arr = getattr(uav_self, attr_name, None)
        pts = set()
        if arr is not None:
            try:
                for p in arr:
                    x, y = int(p[0]), int(p[1])
                    if 0 <= x < H and 0 <= y < W:
                        pts.add((x, y))
            except TypeError:
                pass
        return np.array(list(pts), dtype=np.int32) if pts else None

    # 仅依据“自己观测到的”火势地图定义三类集合
    P_head  = _collect_points_for_self("B")
    P_flank = _collect_points_for_self("B_flank")
    P_tail  = _collect_points_for_self("B_tail")
    idx_head  = [i for i in others_idx if int(self.uavs[i].option) == 1]
    idx_flank = [i for i in others_idx if int(self.uavs[i].option) == 2]
    idx_tail  = [i for i in others_idx if int(self.uavs[i].option) == 3]
    # print(f"选了头部的序号为{idx_head}")
    # print(f"选了侧翼的序号为{idx_flank}")
    # print(f"选了尾巴的序号为{idx_tail}")
    eff_h = _eff_count(idx_head)
    eff_f = _eff_count(idx_flank)
    eff_t = _eff_count(idx_tail)

    has_head  = P_head  is not None and len(P_head)  > 0
    has_flank = P_flank is not None and len(P_flank) > 0
    has_tail  = P_tail  is not None and len(P_tail)  > 0

    need_head  = 0.0 if not has_head  else max(0.0, float(target_staffing[0]) - eff_h)
    need_flank = 0.0 if not has_flank else max(0.0, float(target_staffing[1]) - eff_f)
    need_tail  = 0.0 if not has_tail  else max(0.0, float(target_staffing[2]) - eff_t)

    relay_need = np.array([need_head, need_flank, need_tail], dtype=np.float32)
    relay_need = relay_need /10
    # print(f"需要程度为{relay_need}")
    # ---------- 3) 队友到三类目标的最短距离（对角线归一化） ----------

    BIG = float(H + W)

    def _team_min_dist(P):
        """
        与 high-obs 自身一致的度量：
        - 曼哈顿距离（L1）
        - 无目标/无队友 => 兜底为 H+W
        - 最后除以 10 做同尺度缩放
        """
        if P is None or K == 0 or len(P) == 0:
            return BIG / 10.0

        # 确保是 (N,2) 的整数数组
        P_arr = np.asarray(P, dtype=np.int32)

        best = BIG
        for i in others_idx:
            vx = int(getattr(self.uavs[i], "x", 0))
            vy = int(getattr(self.uavs[i], "y", 0))
            # 曼哈顿距离：|x - vx| + |y - vy|
            dmin = np.abs(P_arr - np.array([vx, vy], dtype=np.int32)).sum(axis=1).min()
            if dmin < best:
                best = float(dmin)

        return best / 10.0

    dmin_head  = _team_min_dist(P_head)
    dmin_flank = _team_min_dist(P_flank)
    dmin_tail  = _team_min_dist(P_tail)
    dmins = np.array([dmin_head, dmin_flank, dmin_tail], dtype=np.float32)

    # ---------- 拼装：4 + 3 + 3 = 10 维 ----------
    return np.concatenate([task_ratio, relay_need, dmins]).astype(np.float32)

# ============== 3) 低层 -> 高层 观测重构 ==============
def reconstruct_high_obs(
    obs_low: np.ndarray,
    idx :int,
    env: GridEnv,
    use_topology_obs: bool = True,
    **kwargs  # 兼容遗留的count参数等
) -> np.ndarray:
    """
    重构高层观测
    
    参数:
        use_topology_obs: 是否使用拓扑观测（用于消融实验）
        **kwargs: 兼容遗留参数（如count，已不再使用）
    """
    import numpy as np

    # 个体特征
    pos  = obs_low[15:17]      # 归一化位置 [x/50, y/50]
    ammo = obs_low[17:18]      # 归一化弹药
    elec = obs_low[18:19]

    # 全局+局部混合特征（风/覆盖率等全局；火头/侧翼/尾部长度用本机集合）
    g = compute_global_features(env, idx, use_topology_obs=use_topology_obs)  # e.g. [wind, coverage, fire_diff, head_len, ...]

    # === 最近火头距离（用 pos 反推栅格坐标，取整）===
    H, W = env.grid["burning_state"].shape
    uav_x = int(round(float(pos[0]) * H))
    uav_y = int(round(float(pos[1]) * W))

    # 火头集合：优先用“当前无人机自己的” uav.B；为空则退化为 burning_state==1 的格子
    uav_i = env.uavs[idx]
    if hasattr(uav_i, "B") and isinstance(uav_i.B, (list, tuple)) and len(uav_i.B) > 0:
        fire_cells = list(uav_i.B)
    else:
        xs, ys = np.where(np.isclose(env.grid["burning_state"], 1.0))
        fire_cells = list(zip(xs.tolist(), ys.tolist()))
    #侧翼集合（同样使用本机的集合）
    if hasattr(uav_i, "B_flank") and isinstance(uav_i.B_flank, (list, tuple)) and len(uav_i.B_flank) > 0:
        fire_cells_flank = list(uav_i.B_flank)
    else:
        xs, ys = np.where(np.isclose(env.grid["burning_state"], 1.0))
        fire_cells_flank = list(zip(xs.tolist(), ys.tolist()))
    #尾部集合（同样使用本机的集合）
    if hasattr(uav_i, "B_tail") and isinstance(uav_i.B_tail, (list, tuple)) and len(uav_i.B_tail) > 0:
        fire_cells_tail = list(uav_i.B_tail)
    else:
        xs, ys = np.where(np.isclose(env.grid["burning_state"], 1.0))
        fire_cells_tail = list(zip(xs.tolist(), ys.tolist()))

    # 计算最近火头距离，若无火点则置为一个大值（比如 H+W）
    dis2fire_val = env._nearest_fire_dist(uav_x, uav_y, fire_cells)
    if not np.isfinite(dis2fire_val):
        dis2fire_val = float(H + W)
    dis2fire_flank = env._nearest_fire_dist(uav_x, uav_y, fire_cells_flank)
    if not np.isfinite(dis2fire_flank):
        dis2fire_val = float(H + W)
    dis2fire_tail = env._nearest_fire_dist(uav_x, uav_y, fire_cells_tail)
    if not np.isfinite(dis2fire_tail):
        dis2fire_val = float(H + W)

    # 包装成一维数组，避免 concatenate 维度不一致
    dis2fire = np.array([dis2fire_val], dtype=np.float32)
    dis2fire_flank = np.array([dis2fire_flank], dtype=np.float32)
    dis2fire_tail = np.array([dis2fire_tail], dtype=np.float32)

    team_vec = build_team_aggregate(self=env,self_idx=idx,prev_options=env.option_counts)
    # print(f"聚合特征为{team_vec}")
    
    # === 计算人群中心距离特征 ===
    # 1. 计算人群中心（所有救援人员的中心位置）
    # 更新逻辑：按不同位置去重后求平均，而不是按数量加权
    # 例如：9个在(0,0)，1个在(10,10)，质点应该是(5,5)而不是(1,1)
    rescuer_center_x = 0.0
    rescuer_center_y = 0.0
    if hasattr(env, "rescuers") and len(env.rescuers) > 0:
        # 收集所有不同的位置（四舍五入到整数，然后去重）
        unique_positions = set()
        for rescuer in env.rescuers:
            # 将位置四舍五入到整数，用于去重
            pos_key = (int(round(rescuer.x)), int(round(rescuer.y)))
            unique_positions.add(pos_key)
        
        # 计算所有不同位置的平均值
        if len(unique_positions) > 0:
            for px, py in unique_positions:
                rescuer_center_x += px
                rescuer_center_y += py
            rescuer_center_x /= len(unique_positions)
            rescuer_center_y /= len(unique_positions)
    
    # 确保num_rescuers变量存在（用于后续判断）
    num_rescuers = len(env.rescuers) if hasattr(env, "rescuers") else 0
    
    # 2. 计算自己到人群中心的距离（曼哈顿距离，归一化）
    if num_rescuers > 0:
        self_dist_to_rescuer_center = abs(uav_x - rescuer_center_x) + abs(uav_y - rescuer_center_y)
        self_dist_to_rescuer_center = self_dist_to_rescuer_center / 10.0  # 归一化
    else:
        self_dist_to_rescuer_center = float(H + W) / 10.0  # 无救援人员时使用大值
    
    # 3. 计算无人机群（包括自己）到人群中心的最短距离
    if num_rescuers > 0:
        min_team_dist_to_rescuer_center = float(H + W)
        for uav in env.uavs:
            uav_x_pos = int(getattr(uav, "x", 0))
            uav_y_pos = int(getattr(uav, "y", 0))
            dist = abs(uav_x_pos - rescuer_center_x) + abs(uav_y_pos - rescuer_center_y)
            if dist < min_team_dist_to_rescuer_center:
                min_team_dist_to_rescuer_center = dist
        min_team_dist_to_rescuer_center = min_team_dist_to_rescuer_center / 10.0  # 归一化
    else:
        min_team_dist_to_rescuer_center = float(H + W) / 10.0  # 无救援人员时使用大值
    
    # 包装成数组
    self_dist_to_rescuer_center = np.array([self_dist_to_rescuer_center], dtype=np.float32)
    min_team_dist_to_rescuer_center = np.array([min_team_dist_to_rescuer_center], dtype=np.float32)
    
    # === 计算高层观测所需的关键字段 ===
    # 1. battery_level（从elec获取，已经是归一化的）
    battery_level = float(elec[0])
    
    # 2. coverage_rate（从全局特征g中提取，g[6]是coverage_rate）
    coverage_rate = float(g[6]) if len(g) > 6 else 0.0
    
    # 3. 上一个时间步的团队任务选择占比（3维：探索、通信、补给）
    # 使用env.option_counts（上一个时间步的任务计数）
    n_agents = len(env.uavs)
    if n_agents > 0 and hasattr(env, "option_counts") and env.option_counts is not None:
        # 计算任务占比（归一化到[0, 1]）
        task_counts = env.option_counts.copy()
        total_tasks = float(np.sum(task_counts)) if np.sum(task_counts) > 0 else 1.0
        # 确保有3个任务类型（探索=0, 通信=1, 补给=2）
        task_ratio_explore = float(task_counts[0]) / total_tasks if 0 < len(task_counts) else 0.0
        task_ratio_comm = float(task_counts[1]) / total_tasks if 1 < len(task_counts) else 0.0
        task_ratio_recharge = float(task_counts[2]) / total_tasks if 2 < len(task_counts) else 0.0
    else:
        # 如果没有历史数据，使用均匀分布
        task_ratio_explore = 1.0 / 3.0
        task_ratio_comm = 1.0 / 3.0
        task_ratio_recharge = 1.0 / 3.0
    
    # 包装成数组
    battery_level_arr = np.array([battery_level], dtype=np.float32)
    coverage_rate_arr = np.array([coverage_rate], dtype=np.float32)
    task_ratio_arr = np.array([task_ratio_explore, task_ratio_comm, task_ratio_recharge], dtype=np.float32)
    

    high_obs = np.concatenate([
        np.asarray(g, dtype=np.float32),  # g[0-6]: wind_speed, rate, fire_diff, head_len, flank_len, tail_len, coverage_rate (7维)
        battery_level_arr,                 # 索引7: battery_level
        coverage_rate_arr,                  # 索引8: coverage_rate (与g[6]重复，但保留用于兼容)
        self_dist_to_rescuer_center,       # 索引9: self_dist_to_rescuer_center
        min_team_dist_to_rescuer_center,   # 索引10: min_team_dist_to_rescuer_center
        task_ratio_arr                     # 索引11-13: 上一个时间步的任务占比（探索、通信、补给）
    ], axis=0).astype(np.float32)

    return high_obs


# ============== 4) option -> 启发式动作 ==============
def option_to_action(option: int, agent_index: int, obs_low: np.ndarray, env: GridEnv) -> int:
    """
    把高层 option（子任务）映射为具体低层离散动作（0..8）
    简化为3个任务：
    0: FireStructureSensing（火情探索）
    1: CommSupport（通信保障）
    2: Recharge（补给）
    """
    uav = env.uavs[agent_index]
    select_band_and_boundary(uav)
    select_flank_by_ybands(uav, 1)
    select_tail_by_xband(uav, 1)
    
    if option == 0:
        # 任务0：火情探索（FireStructureSensing）- 复用现有的探索逻辑
        return heuristic_action_using_map(uav, obs_low)
    elif option == 1:
        # 任务1：通信保障（CommSupport）- 朝向未被覆盖的救援人员
        return step_comm_support(uav, env)
    elif option == 2:
        # 任务2：补给（Recharge）- 复用现有的充电逻辑
        return step_go_charge(uav, env)
    else:
        return 8  # 未定义则"原地"
    
def any_uav_at_station(env, tol=0):
    """只要有任意 UAV 处在任一充电/补给点上，就返回 True"""
    stations = set(getattr(env, "charging_stations", []))
    for u in env.uavs:
        px = int(round(u.x)); py = int(round(u.y))
        if (px, py) in stations:
            return True
    return False


def test_trained_model(model_path="Result/model/checkpoint_ep1000.pth",
                       n_episodes=5, max_steps=200, n_options=3,
                       use_topology_obs: bool = True, use_topology_exploration: bool = True):
    """
    测试训练好的模型
    
    参数:
        use_topology_obs: 是否使用拓扑观测（必须与训练时一致）
        use_topology_exploration: 是否使用拓扑感知的探索调度（评估时会被忽略，epsilon=0）
    """
    import numpy as np
    import torch

    env = GridEnv()
    obs_dim = 14  # 与训练一致
    policy = HighLevelPolicy(
        n_options=n_options, 
        obs_dim=obs_dim,
        use_topology_obs=use_topology_obs,
        use_topology_exploration=use_topology_exploration
    )  # 与训练一致

    # ==== 加载模型 ====
    checkpoint = torch.load(model_path, map_location=policy.device)
    policy.online.load_state_dict(checkpoint["online_state_dict"])
    policy.target.load_state_dict(checkpoint["target_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        policy.opt.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"[Test] 模型已加载 {model_path}")

    returns = []
    K = 5  # 和训练一致

    for ep in range(n_episodes):
        obs_list_low, _ = env.reset()
        n_agents = len(env.uavs)

        ep_ret = np.zeros(n_agents, dtype=np.float32)
        terminated = truncated = False  # 防止未定义
        t = 0

        while t < max_steps:
            # ===== 1) 高层决策（使用select_greedy_masked） =====
            options_this_step = []
            high_obs_list = []
            option_counts = np.zeros(n_options, dtype=np.int32)

            for i, obs_low in enumerate(obs_list_low):
                # 构建高层观测（与训练一致，不使用count参数）
                high_obs = reconstruct_high_obs(obs_low, i, env, use_topology_obs=policy.use_topology_obs)
                
                # 使用select_greedy_masked进行决策（带掩码的贪心选择，评估时epsilon=0）
                with torch.no_grad():
                    option = policy.select_greedy_masked(high_obs)
                
                env.uavs[i].option = option
                option_counts[option] += 1
                options_this_step.append(option)
                high_obs_list.append(high_obs)

            env.option_counts = option_counts.copy()

            # ===== 2) 低层执行 K 步（或直到终止/事件触发提前重选） =====
            k = 0
            while k < K and t < max_steps:
                actions = []
                for i, obs_low in enumerate(obs_list_low):
                    # 由高层 option 生成低层动作
                    act = option_to_action(options_this_step[i], i, obs_low, env)
                    actions.append(act)

                next_obs_list_low, rewards, terminated, truncated, infos = env.step(actions)
                env.render()
                
                # === 事件检测先于奖励累计（与训练一致） ===
                at_station = any_uav_at_station(env)

                if at_station:
                    # 触发提前重选：本步不累计奖励
                    obs_list_low = next_obs_list_low
                    t += 1
                    k += 1
                    break  # 立刻跳出 K 循环，回到外层进行新一轮高层决策

                # —— 未触发事件：才累计本步奖励
                ep_ret += np.array(rewards, dtype=np.float32)

                obs_list_low = next_obs_list_low
                t += 1
                k += 1

                if terminated or truncated:
                    break
                    
            if terminated or truncated:
                break  # 结束 episode
        # print(f"过火面积:{int(np.count_nonzero(env.grid['burning_state'] >0))}")

        mean_ret = ep_ret.mean()
        returns.append(mean_ret)
        print(f"[Test EP {ep+1}] mean return/agent = {mean_ret:.3f}, steps = {t}")
        # print(f"过火面积:{int(np.count_nonzero(env.grid['burning_state'] >0))}")
    print(f"[Test Summary] 平均回报 = {np.mean(returns):.3f} ± {np.std(returns):.3f}")


def visualize_trained_model(model_path: str,
                           max_steps: int = 150,
                           n_options: int = 3,
                           use_topology_obs: bool = True,
                           use_topology_exploration: bool = True,
                           pause_time: float = 0.1):
    """
    加载训练好的模型，运行一轮episode并可视化
    
    参数:
        model_path: 模型文件路径
        max_steps: 最大步数
        n_options: 任务选项数量
        use_topology_obs: 是否使用拓扑观测（必须与训练时一致）
        use_topology_exploration: 是否使用拓扑感知的探索调度（评估时会被忽略）
        pause_time: 渲染时的暂停时间（秒），用于控制动画速度
    
    返回:
        dict: 包含总奖励、步数、覆盖率等统计信息
    """
    import numpy as np
    import torch
    import time
    
    print("=" * 60)
    print("开始可视化训练好的模型")
    print(f"模型路径: {model_path}")
    print(f"最大步数: {max_steps}")
    print("=" * 60)
    
    # 初始化环境（使用human模式以便可视化）
    env = GridEnv(render_mode="human")
    obs_dim = 14  # 与训练一致
    policy = HighLevelPolicy(
        n_options=n_options, 
        obs_dim=obs_dim,
        use_topology_obs=use_topology_obs,
        use_topology_exploration=use_topology_exploration
    )
    
    # ==== 加载模型 ====
    try:
        checkpoint = torch.load(model_path, map_location=policy.device)
        policy.online.load_state_dict(checkpoint["online_state_dict"])
        policy.target.load_state_dict(checkpoint["target_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            policy.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[加载成功] 模型已从 {model_path} 加载")
    except Exception as e:
        print(f"[错误] 无法加载模型: {e}")
        return None
    
    # 初始化
    obs_list_low, _ = env.reset()
    n_agents = len(env.uavs)
    K = 5  # 和训练一致
    
    ep_ret = np.zeros(n_agents, dtype=np.float32)
    t = 0
    
    # 统计信息
    coverage_rates = []
    explore_rates = []
    option_history = []  # 记录每个时间步的任务选择
    
    # 记录每一步的奖励数据（用于保存CSV）
    step_reward_data = []  # 列表，每个元素是一个字典，包含该步的所有信息
    
    print(f"\n开始运行episode（最大{max_steps}步）...")
    print("-" * 60)
    
    while t < max_steps:
        # ===== 1) 高层决策（使用select_greedy_masked） =====
        options_this_step = []
        high_obs_list = []
        option_counts = np.zeros(n_options, dtype=np.int32)
        
        for i, obs_low in enumerate(obs_list_low):
            # 构建高层观测
            high_obs = reconstruct_high_obs(obs_low, i, env, use_topology_obs=policy.use_topology_obs)
            
            # 使用select_greedy_masked进行决策（评估时epsilon=0）
            with torch.no_grad():
                option = policy.select_greedy_masked(high_obs)
            
            env.uavs[i].option = option
            option_counts[option] += 1
            options_this_step.append(option)
            high_obs_list.append(high_obs)
        
        env.option_counts = option_counts.copy()
        option_history.append(options_this_step.copy())
        
        # ===== 2) 低层执行 K 步（或直到终止/事件触发提前重选） =====
        k = 0
        while k < K and t < max_steps:
            actions = []
            for i, obs_low in enumerate(obs_list_low):
                # 由高层 option 生成低层动作
                act = option_to_action(options_this_step[i], i, obs_low, env)
                actions.append(act)
            
            next_obs_list_low, rewards, terminated, truncated, infos = env.step(actions)
            
            # ===== 渲染可视化 =====
            env.render()
            time.sleep(pause_time)  # 控制动画速度
            
            # 记录统计信息
            coverage_rate = float(getattr(env, "coverage_rate", 0.0))
            true_b = max(1, int(getattr(env, "true_boundary", 0)))
            known_b = int(getattr(env, "known_boundary", 0))
            explore_rate = float(known_b) / float(true_b) if true_b > 0 else 0.0
            
            coverage_rates.append(coverage_rate)
            explore_rates.append(explore_rate)
            
            # ===== 输出当前步的任务状态和奖励 =====
            task_names = {0: "探索", 1: "通信", 2: "补给"}
            print(f"\n[步数 {t+1}] 低层步 {k+1}/{K}")
            print(f"  全局状态: 覆盖率={coverage_rate:.3f}, 探索率={explore_rate:.3f}")
            
            # 记录当前步的数据
            step_data = {
                "步数": t + 1,
                "低层步": k + 1,
                "覆盖率": coverage_rate,
                "探索率": explore_rate,
                "总奖励": float(sum(rewards))
            }
            
            for i, (uav, opt, reward) in enumerate(zip(env.uavs, options_this_step, rewards)):
                task_name = task_names.get(opt, f"未知({opt})")
                battery = float(getattr(uav, "e", 1000.0)) / 1000.0
                ammo = float(getattr(uav, "d", 0.0))
                pos_x, pos_y = int(uav.x), int(uav.y)
                
                # 记录每个无人机的信息
                step_data[f"无人机{i}_任务"] = task_name
                step_data[f"无人机{i}_任务编号"] = opt
                step_data[f"无人机{i}_奖励"] = float(reward)
                step_data[f"无人机{i}_位置X"] = pos_x
                step_data[f"无人机{i}_位置Y"] = pos_y
                step_data[f"无人机{i}_电量"] = battery
                step_data[f"无人机{i}_弹药"] = ammo
                
                print(f"  无人机{i}: 任务={task_name}({opt}) | "
                      f"奖励={reward:.3f} | "
                      f"位置=({pos_x},{pos_y}) | "
                      f"电量={battery:.2f} | "
                      f"弹药={ammo:.1f}")
            print(f"  总奖励(本步): {sum(rewards):.3f}")
            
            # 添加到记录列表
            step_reward_data.append(step_data)
            
            # === 事件检测 ===
            at_station = any_uav_at_station(env)
            
            if at_station:
                # 触发提前重选：本步不累计奖励
                obs_list_low = next_obs_list_low
                t += 1
                k += 1
                break  # 立刻跳出 K 循环，回到外层进行新一轮高层决策
            
            # 累计奖励
            ep_ret += np.array(rewards, dtype=np.float32)
            
            obs_list_low = next_obs_list_low
            t += 1
            k += 1
            
            # 已删除环境层面的终止条件，只通过max_steps控制
            # if terminated or truncated:
            #     break
                
        # 打印当前进度
        if t % 10 == 0:
            print(f"步数: {t}/{max_steps} | "
                  f"总奖励: {ep_ret.mean():.3f} | "
                  f"覆盖率: {coverage_rate:.3f} | "
                  f"探索率: {explore_rate:.3f}")
    
    # ===== 统计结果 =====
    total_reward = ep_ret.mean()
    avg_coverage = np.mean(coverage_rates) if coverage_rates else 0.0
    avg_explore = np.mean(explore_rates) if explore_rates else 0.0
    final_coverage = coverage_rates[-1] if coverage_rates else 0.0
    final_explore = explore_rates[-1] if explore_rates else 0.0
    
    # 统计任务选择分布
    task_names = {0: "探索", 1: "通信", 2: "补给"}
    task_counts = {0: 0, 1: 0, 2: 0}
    for step_options in option_history:
        for opt in step_options:
            if opt in task_counts:
                task_counts[opt] += 1
    
    print("\n" + "=" * 60)
    print("Episode 运行完成！")
    print("=" * 60)
    print(f"总步数: {t}")
    print(f"总奖励（平均）: {total_reward:.3f}")
    print(f"平均覆盖率: {avg_coverage:.4f}")
    print(f"最终覆盖率: {final_coverage:.4f}")
    print(f"平均探索率: {avg_explore:.4f}")
    print(f"最终探索率: {final_explore:.4f}")
    print("\n任务选择统计:")
    for opt, count in task_counts.items():
        percentage = (count / sum(task_counts.values()) * 100) if sum(task_counts.values()) > 0 else 0
        print(f"  {task_names[opt]}: {count} 次 ({percentage:.1f}%)")
    print("=" * 60)
    
    # ===== 保存奖励数据到CSV =====
    csv_filename = None
    if step_reward_data:
        import os
        import pandas as pd
        import datetime
        
        # 确保保存目录存在
        save_dir = "hrl_dqn_improve/Result_v1"
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建DataFrame
        df_rewards = pd.DataFrame(step_reward_data)
        
        # 生成CSV文件名（包含模型名称和时间戳）
        model_name = os.path.basename(model_path).replace(".pth", "")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{save_dir}/step_rewards.csv"
        
        # 保存CSV
        df_rewards.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"\n[CSV保存] 每步奖励数据已保存到: {csv_filename}")
        print(f"  共记录 {len(step_reward_data)} 步数据")
    
    # 返回统计信息
    results = {
        "total_steps": t,
        "total_reward": float(total_reward),
        "avg_coverage": float(avg_coverage),
        "final_coverage": float(final_coverage),
        "avg_explore": float(avg_explore),
        "final_explore": float(final_explore),
        "task_distribution": task_counts,
        "coverage_history": coverage_rates,
        "explore_history": explore_rates,
        "step_reward_csv": csv_filename
    }
    
    return results


# ============== 5) 高层训练主循环骨架（无网络，仅数据流） ==============
def train_high_level(
    num_episodes: int = 10000,
    max_steps: int = 200,
    n_options: int = 3,
    resume_from: str = None,
    use_topology_obs: bool = True,
    use_topology_exploration: bool = True,
):
    """
    训练高层策略
    
    参数:
        use_topology_obs: 是否使用拓扑观测（用于消融实验）
        use_topology_exploration: 是否使用拓扑感知的探索调度
    """
    K = 10  # 每次高层决策持续的低层步数
    env = GridEnv()
    obs_dim = 14  # g(7) + battery_level(1) + coverage_rate(1) + self_dist(1) + min_team_dist(1) + task_ratio(3)
    policy = HighLevelPolicy(
        n_options=n_options, 
        obs_dim=obs_dim,
        use_topology_obs=use_topology_obs,
        use_topology_exploration=use_topology_exploration
    )
    episode_rewards = []
    losses = []
    # 新增：记录每个episode的平均覆盖率和第一次达成100%探索的时间步
    episode_avg_coverage = []
    episode_avg_explore = []
    first_100pct_explore_step = []
    # 新增：记录通感融合指标（平均探索率*0.5 + 平均覆盖率*0.5）
    fusion_metrics = []
    episode_max_aoi = []
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=policy.device)
        policy.online.load_state_dict(ckpt["online_state_dict"])
        policy.target.load_state_dict(ckpt["target_state_dict"])
        policy.opt.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[Resume] 模型已加载 {resume_from}")
    else:
        if resume_from:
            print(f"[Resume] 未找到文件: {resume_from},开始随机训练")


    
    for ep in range(num_episodes):
        obs_list_low, _ = env.reset()
        n_agents = len(env.uavs)
        
        # 初始化覆盖率历史（每集开始时重置）
        env._coverage_rate_history = []
        
        # 记录本episode每个高层步的覆盖率和探索率
        episode_coverage_list = []
        episode_explore_list = []
        # 记录第一次达成100%探索的时间步（如果达成）
        first_100pct_step = None

        ep_ret = np.zeros(n_agents, dtype=np.float32)

        t = 0
        while t < max_steps:
            # ===== 1) 选一次高层 option（基于当前高层观测） =====
            options_this_step: List[int] = []
            high_obs_list: List[np.ndarray] = []

            option_counts = np.zeros(n_options, dtype=np.int32)

            for i, obs_low in enumerate(obs_list_low):


                high_obs = reconstruct_high_obs(obs_low, i, env, use_topology_obs=policy.use_topology_obs)
    

                option = policy.select_option(high_obs, ep, env=env, uav_idx=i, is_eval=False)
                env.uavs[i].option = option

                option_counts[option] += 1
                options_this_step.append(option)
                high_obs_list.append(high_obs)


            #四个任务的选取情况
            env.option_counts = option_counts.copy()

            # ===== 2) 低层执行 K 步（或直到终止/事件触发提前重选） =====
            reward_sum = np.zeros(n_agents, dtype=np.float32)
            k = 0
            disc = 1.0
            gamma = policy.gamma

            while k < K and t < max_steps:
                # 固定 option 生成动作
                mem_start = [int(obs[0]) for obs in obs_list_low]
                actions: List[int] = []
                for i, obs_low in enumerate(obs_list_low):
                    act = option_to_action(options_this_step[i], i, obs_low, env)
                    actions.append(act)

                next_obs_list_low, rewards, terminated, truncated, infos = env.step(actions)
                # env.render()
                # === 事件检测先于奖励累计（与 prefill 对齐） ===

                
                at_station = any_uav_at_station(env)

                if at_station:
                    # 触发提前重选：本步不累计奖励、不推进折扣
                    obs_list_low = next_obs_list_low
                    t += 1
                    k += 1
                    break  # 立刻跳出 K 循环，回到外层进行新一轮高层决策

                # —— 未触发事件：才累计本步奖励与折扣
                reward_sum += disc * np.array(rewards, dtype=np.float32)
                disc *= gamma

                ep_ret += np.array(rewards, dtype=np.float32)

                obs_list_low = next_obs_list_low
                t += 1
                k += 1

                # 已删除环境层面的终止条件，不再检查terminated或truncated提前结束
                # 只通过max_steps控制episode长度
                # if terminated or truncated:
                #     break

            # ===== 3) 形成一次"高层 transition"（K步聚合）=====
            next_high_obs_list = [
                reconstruct_high_obs(obs_low, i, env, use_topology_obs=policy.use_topology_obs)
                for i in range(n_agents)
            ]
            # 已删除环境层面的终止条件，done_flag始终为False，只通过max_steps控制结束
            done_flag = False

            # ===== 输出每个无人机的任务类型和奖励 =====
            # 计算覆盖率和探索率
            coverage_rate = float(getattr(env, "coverage_rate", 0.0))
            true_b = max(1, int(getattr(env, "true_boundary", 0)))
            known_b = int(getattr(env, "known_boundary", 0))
            explore_rate = float(known_b) / float(true_b) if true_b > 0 else 0.0
            
            # 记录当前高层步的覆盖率和探索率
            episode_coverage_list.append(coverage_rate)
            episode_explore_list.append(explore_rate)
            
            # 检查是否第一次达成100%探索
            if first_100pct_step is None and explore_rate >= 0.98:
                first_100pct_step = t
            

            for i in range(n_agents):
                opt = options_this_step[i]
                transition = {
                    "obs_high":      high_obs_list[i],
                    "option":        options_this_step[i],
                    "Rk":            float(reward_sum[i]),
                    "next_obs_high": next_high_obs_list[i],
                    "done":          done_flag,
                }
                policy.push(transition)
            loss = policy.update()
            if loss is not None:
                losses.append(loss)

            # 已删除提前结束检查，只通过max_steps控制episode长度
            # if done_flag:
            #     break  # 结束 episode
            # print("---------------------------------")

        # 一集结束
        total_reward = ep_ret.mean()
        episode_rewards.append(total_reward)
        
        # 计算本episode的平均覆盖率和平均探索率
        avg_coverage_ep = np.mean(episode_coverage_list) if episode_coverage_list else 0.0
        avg_explore_ep = np.mean(episode_explore_list) if episode_explore_list else 0.0
        
        episode_avg_coverage.append(avg_coverage_ep)
        episode_avg_explore.append(avg_explore_ep)

        # 计算通感融合指标：平均探索率*0.5 + 平均覆盖率*0.5
        fusion_metric = avg_explore_ep * 0.5 + avg_coverage_ep * 0.5
        fusion_metrics.append(fusion_metric)
        
        # 记录本episode的最大AoI
        current_max_aoi = float(getattr(env, "max_AoI", 0.0))
        episode_max_aoi.append(current_max_aoi)

        # 输出每一轮的总奖励和通感融合指标
        print(f"[EP {ep:04d}] 总奖励 = {total_reward:.3f}, 步数 = {t}, "
              f"平均探索率 = {avg_explore_ep:.3f}, 平均覆盖率 = {avg_coverage_ep:.3f}, "
              f"通感融合指标 = {fusion_metric:.3f}")
        
        # ===== 每episode保存CSV文件 =====
        if (ep + 1) % 100 == 0:
            # 确保保存目录存在
            os.makedirs("hrl_dqn_improve/Result_without_qiehuan/reward", exist_ok=True)
            
            # 准备数据：将列表组合成DataFrame
            # 确保所有列表长度一致
            n_episodes = len(episode_rewards)
            data = {
                "episode": list(range(n_episodes)),
                "reward": episode_rewards,
            }
            
            # 如果通感融合指标列表长度匹配，也添加到CSV中
            if len(fusion_metrics) == n_episodes:
                data["fusion_metric"] = fusion_metrics
            # 添加平均覆盖率和探索率列
            if len(episode_avg_coverage) == n_episodes:
                data["avg_coverage"] = episode_avg_coverage
            if len(episode_avg_explore) == n_episodes:
                data["avg_explore"] = episode_avg_explore
            # 添加最大AoI列
            if len(episode_max_aoi) == n_episodes:
                data["max_aoi"] = episode_max_aoi
            
            df = pd.DataFrame(data)
            csv_path = f"hrl_dqn_improve/Result_without_qiehuan/reward/training_data.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"[CSV] 训练数据已保存到 {csv_path} (共{len(df)}条记录)")


        if (ep+1)%50 == 0 :
            # 确保保存目录存在
            os.makedirs("hrl_dqn_improve/Result_without_qiehuan/pic", exist_ok=True)
            
            # 保存Reward曲线（平滑处理）
            smoothed_reward = moving_average(episode_rewards, window_size=50)
            plt.figure(figsize=(10, 6))
            plt.plot(episode_rewards, alpha=0.3, label="Raw Reward")
            plt.plot(smoothed_reward, color='orange', label="Smoothed Reward")
            plt.xlabel("Episode")
            plt.ylabel("Mean Return / Agent")
            plt.title("Training Reward Curve")
            plt.legend()
            plt.grid(True)
            plt.savefig("hrl_dqn_improve/Result_without_qiehuan/pic/reward_topology_curve.png", dpi=150)
            plt.close()

            # 保存Q-Loss曲线
            plt.figure(figsize=(10, 6))
            plt.plot(losses)
            plt.xlabel("Update Step")
            plt.ylabel("Q Loss")
            plt.title("Q-Loss Curve")
            plt.grid(True)
            plt.savefig("hrl_dqn_improve/Result_without_qiehuan/pic/q_loss.png", dpi=150)
            plt.close()
            
            if len(fusion_metrics) > 0:
                plt.figure(figsize=(10, 6))
                episodes_list = list(range(len(fusion_metrics)))

                # 原始曲线
                plt.plot(episodes_list, fusion_metrics, color='green', alpha=0.3, label="Raw Fusion Metric")
                if len(episode_avg_explore) == len(episodes_list):
                    plt.plot(episodes_list, episode_avg_explore, color='blue', alpha=0.3, label="Avg Explore Rate")
                if len(episode_avg_coverage) == len(episodes_list):
                    plt.plot(episodes_list, episode_avg_coverage, color='red', alpha=0.3, label="Avg Coverage Rate")

                # 平滑处理
                try:
                    smoothed_fusion = moving_average(fusion_metrics, window_size=50)
                    smooth_aoi = moving_average(episode_max_aoi, window_size=50)
                except Exception:
                    smoothed_fusion = fusion_metrics
                plt.plot(smoothed_fusion, color='green', linewidth=2, label="Smoothed Fusion (w=50)")

                # 对探索率和覆盖率也进行平滑
                if len(episode_avg_explore) == len(episodes_list):
                    try:
                        smooth_explore = moving_average(episode_avg_explore, window_size=50)
                    except Exception:
                        smooth_explore = episode_avg_explore
                    plt.plot(smooth_explore, color='blue', linewidth=2, label="Smoothed Explore (w=50)")
                if len(episode_avg_coverage) == len(episodes_list):
                    try:
                        smooth_cov = moving_average(episode_avg_coverage, window_size=50)
                    except Exception:
                        smooth_cov = episode_avg_coverage
                    plt.plot(smooth_cov, color='red', linewidth=2, label="Smoothed Coverage (w=50)")

                plt.xlabel("训练轮数 (Episode)")
                plt.ylabel("比率 / 指标")
                plt.title("通感融合指标与平均探索/覆盖率曲线")
                plt.legend()
                plt.grid(True)
                plt.savefig("hrl_dqn_improve/Result_without_qiehuan/pic/e&c.png", dpi=150)
                plt.close()
                print(f"[图像] 通感融合指标曲线已更新 (共{len(fusion_metrics)}个数据点)")

            plt.figure(figsize=(10, 6))
            plt.plot(episode_max_aoi, color='blue', alpha=0.3, label="Raw Max AoI")
            plt.plot(smooth_aoi, color='orange', linewidth=2, label="Smoothed Max AoI (w=50)")
            plt.xlabel("Episode")
            plt.ylabel("Max AoI")
            plt.title("Max AoI Curve")
            plt.grid(True)
            plt.savefig("hrl_dqn_improve/Result_without_qiehuan/pic/aoi.png", dpi=150)
            plt.close()

        if ep % 500 == 0:
            save_path = f"hrl_dqn_improve/Result_without_qiehuan/new_checkpoint{ep}.pth"
            torch.save({
                "online_state_dict": policy.online.state_dict(),
                "target_state_dict": policy.target.state_dict(),
                "optimizer_state_dict": policy.opt.state_dict(),
                "replay_size": len(policy.replay)
            }, save_path)
            print(f"[Checkpoint] 模型已保存到 {save_path}")





def coverage_explore_experiment(
    model_path="hrl_dqn/Result/model/new_checkpoint_ep3000.pth",
    spread_steps_list=(30, 40, 50),
    num_uavs_list=(2, 4, 6),
    init_hw=(4, 4),
    wind_level=6,
    num_rescuers=8,
    n_options=3,
    max_steps=150,
    K=5,
):
    """
    在固定风级与初始火源(4x4)下，遍历不同扩散步数与无人机数量，共9种组合。
    每种组合下：
      - 强化学习策略跑1次
      - 固定策略跑1次（2架:1探1感知；4架:2探2感知；6架:3探3感知；电量<20%时改为充电 option=2）
    记录每次整轮的平均覆盖率和平均探索率（对时间步求平均），并打印。
    """
    import numpy as np
    import torch

    env = GridEnv()
    policy = HighLevelPolicy(n_options=n_options, obs_dim=14)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=policy.device)
        policy.online.load_state_dict(ckpt["online_state_dict"])
        policy.target.load_state_dict(ckpt["target_state_dict"])
        if "optimizer_state_dict" in ckpt:
            policy.opt.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[CoverageExp] 模型已加载: {model_path}")
    else:
        print(f"[CoverageExp] 警告: 未找到模型 {model_path}，将使用随机初始化策略")

    def _setup_env(spread, num_uav):
        """按给定参数重置环境并返回初始观测列表。"""
        env.wind_level = int(wind_level)
        env.wind_speed = float(wind_level * 2)
        env.spread_steps = int(spread)
        env.init_hw = init_hw
        env.num_uavs = int(num_uav)
        env.num_rescuers = int(num_rescuers)
        obs_list_low, _ = env.reset()
        return obs_list_low

    def _run_rl_episode(spread, num_uav):
        """强化学习策略跑一轮，返回(平均覆盖率, 平均探索率)。"""
        obs_list_low = _setup_env(spread, num_uav)
        t = 0
        terminated = truncated = False
        cov_sum = 0.0
        exp_sum = 0.0
        step_cnt = 0

        while t < max_steps:
            options_this_step = []
            option_counts = np.zeros(n_options, dtype=np.int32)

            for i, obs_low in enumerate(obs_list_low):
                high_obs = reconstruct_high_obs(obs_low, i, env)
                with torch.no_grad():
                    option = policy.select_greedy_masked(high_obs)
                env.uavs[i].option = option
                option_counts[option] += 1
                options_this_step.append(option)

            env.option_counts = option_counts.copy()

            k = 0
            while k < K and t < max_steps:
                actions = [
                    option_to_action(options_this_step[i], i, obs_low, env)
                    for i, obs_low in enumerate(obs_list_low)
                ]
                next_obs_list_low, rewards, terminated, truncated, infos = env.step(actions)

                true_b = max(1, int(getattr(env, "true_boundary", 0)))
                known_b = int(getattr(env, "known_boundary", 0))
                explore_rate = float(known_b) / float(true_b) if true_b > 0 else 0.0
                coverage_rate = float(getattr(env, "coverage_rate", 0.0))

                cov_sum += coverage_rate
                exp_sum += explore_rate
                step_cnt += 1

                at_station = any_uav_at_station(env)

                obs_list_low = next_obs_list_low
                t += 1
                k += 1

                if terminated or truncated or at_station:
                    break

            if terminated or truncated:
                break

        avg_cov = cov_sum / step_cnt if step_cnt > 0 else 0.0
        avg_exp = exp_sum / step_cnt if step_cnt > 0 else 0.0
        return float(avg_cov), float(avg_exp)

    def _build_fixed_base(num_uav: int):
        """构造固定策略的基础任务分配（不含充电）。"""
        if num_uav == 2:
            base = [0, 1]  # 1探1感知
        elif num_uav == 4:
            base = [0, 0, 1, 1]  # 2探2感知
        elif num_uav == 6:
            base = [0, 0, 0, 1, 1, 1]  # 3探3感知
        else:
            # 兜底：前一半探索，后一半感知
            k = num_uav // 2
            base = [0] * k + [1] * (num_uav - k)
        return base

    def _run_fixed_episode(spread, num_uav):
        """固定策略跑一轮，返回(平均覆盖率, 平均探索率)。"""
        obs_list_low = _setup_env(spread, num_uav)
        t = 0
        terminated = truncated = False
        cov_sum = 0.0
        exp_sum = 0.0
        step_cnt = 0

        base_options = _build_fixed_base(int(num_uav))
        battery_threshold = 100  # ≈20% 电量（假定满电约1000）

        while t < max_steps:
            n_agents = len(env.uavs)
            options_this_step = base_options[:n_agents].copy()
            option_counts = np.zeros(n_options, dtype=np.int32)

            # 电量低于20%时切换为充电（option=2）
            for i in range(n_agents):
                if getattr(env.uavs[i], "e", 0) < battery_threshold:
                    options_this_step[i] = 2
                env.uavs[i].option = options_this_step[i]
                if 0 <= options_this_step[i] < n_options:
                    option_counts[options_this_step[i]] += 1

            env.option_counts = option_counts.copy()

            k = 0
            while k < K and t < max_steps:
                actions = [
                    option_to_action(options_this_step[i], i, obs_low, env)
                    for i, obs_low in enumerate(obs_list_low)
                ]
                next_obs_list_low, rewards, terminated, truncated, infos = env.step(actions)

                true_b = max(1, int(getattr(env, "true_boundary", 0)))
                known_b = int(getattr(env, "known_boundary", 0))
                explore_rate = float(known_b) / float(true_b) if true_b > 0 else 0.0
                coverage_rate = float(getattr(env, "coverage_rate", 0.0))

                cov_sum += coverage_rate
                exp_sum += explore_rate
                step_cnt += 1

                at_station = any_uav_at_station(env)

                obs_list_low = next_obs_list_low
                t += 1
                k += 1

                if terminated or truncated or at_station:
                    break

            if terminated or truncated:
                break

        avg_cov = cov_sum / step_cnt if step_cnt > 0 else 0.0
        avg_exp = exp_sum / step_cnt if step_cnt > 0 else 0.0
        return float(avg_cov), float(avg_exp)

    for spread in spread_steps_list:
        for num_uav in num_uavs_list:
            rl_cov, rl_exp = _run_rl_episode(spread, num_uav)
            fixed_cov, fixed_exp = _run_fixed_episode(spread, num_uav)

            print(
                f"[CoverageExp] 风级={wind_level}, 扩散步数={spread}, 无人机数={num_uav}\n"
                f"  强化学习: 平均覆盖率={rl_cov:.4f}, 平均探索率={rl_exp:.4f}\n"
                f"  固定策略: 平均覆盖率={fixed_cov:.4f}, 平均探索率={fixed_exp:.4f}"
            )

    # 函数主要用于打印结果，返回值可选
    return None

def comparison_experiment(
    model_path="hrl_dqn/Result/model/new_checkpoint_ep3000.pth",
    max_steps=100,
    n_options=3,
    num_uavs=3,
    save_path="comparison_experiment_results.xlsx"
):
    """
    对比实验函数：比较强化学习策略与固定策略的性能
    
    参数:
        model_path: 强化学习模型路径
        max_steps: 每个实验的最大步数
        n_options: 选项数量（任务类型数）
        num_uavs: 无人机数量（固定为3）
        save_path: 保存Excel文件的路径
    
    实验组：
    - 实验组：强化学习切换策略
    - 对照组1：固定策略 - 1架只做通信覆盖(option=1)，2架只做探索(option=0)
    - 对照组2：固定策略 - 2架只做通信覆盖(option=1)，1架只做探索(option=0)
    
    记录指标：
    - 每一步的探索率（known_boundary / true_boundary）
    - 每一步的覆盖率（coverage_rate）
    """
    import numpy as np
    import torch
    import pandas as pd
    import os
    
    print("=" * 60)
    print("开始对比实验")
    print(f"无人机数量: {num_uavs}")
    print(f"最大步数: {max_steps}")
    print("=" * 60)
    
    # 初始化环境和策略
    env = GridEnv()
    env.num_uavs = num_uavs  # 固定无人机数量为3
    obs_dim = 14
    policy = HighLevelPolicy(n_options=n_options, obs_dim=obs_dim)
    
    # 加载强化学习模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=policy.device)
        policy.online.load_state_dict(checkpoint["online_state_dict"])
        policy.target.load_state_dict(checkpoint["target_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            policy.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[对比实验] 强化学习模型已加载: {model_path}")
    else:
        print(f"[警告] 模型文件不存在: {model_path}，将使用随机初始化的模型")
    
    K = 5  # 高层决策持续的低层步数
    
    # 存储所有实验的数据
    all_data = []
    
    # ========== 实验组：强化学习策略 ==========
    print("\n" + "=" * 60)
    print("实验组：强化学习策略")
    print("=" * 60)
    
    obs_list_low, _ = env.reset()
    n_agents = len(env.uavs)
    terminated = truncated = False
    t = 0
    high_level_step = 0  # 高层步计数器
    
    while t < max_steps:
        high_level_step += 1  # 每个高层决策算一步
        # 高层决策
        options_this_step = []
        high_obs_list = []
        option_counts = np.zeros(n_options, dtype=np.int32)
        
        for i, obs_low in enumerate(obs_list_low):
            high_obs = reconstruct_high_obs(obs_low, i, env)
            with torch.no_grad():
                option = policy.select_greedy_masked(high_obs)
            env.uavs[i].option = option
            option_counts[option] += 1
            options_this_step.append(option)
            high_obs_list.append(high_obs)
        
        env.option_counts = option_counts.copy()
        
        # 低层执行 K 步
        k = 0
        while k < K and t < max_steps:
            actions = []
            for i, obs_low in enumerate(obs_list_low):
                act = option_to_action(options_this_step[i], i, obs_low, env)
                actions.append(act)
            
            next_obs_list_low, rewards, terminated, truncated, infos = env.step(actions)
            # env.render()
            # 在每个低层时间步记录一次数据
            true_b = max(1, int(getattr(env, "true_boundary", 0)))
            known_b = int(getattr(env, "known_boundary", 0))
            explore_rate = float(known_b) / float(true_b) if true_b > 0 else 0.0
            coverage_rate = float(getattr(env, "coverage_rate", 0.0))
            
            all_data.append({
                "时间步": t + 1,  # 低层时间步从1开始
                "实验组": "强化学习",
                "探索率": explore_rate,
                "覆盖率": coverage_rate
            })
            
            at_station = any_uav_at_station(env)
            if at_station:
                obs_list_low = next_obs_list_low
                t += 1
                k += 1
                break
            
            obs_list_low = next_obs_list_low
            t += 1
            k += 1
            
            if terminated or truncated:
                # 对比实验不提前结束，仅跳出当前K循环
                break
        
        # 对比实验不提前结束，只有到达max_steps才结束
        terminated = False
        truncated = False
    
    print(f"实验组完成，总步数: {t}")
    
    # ========== 对照组1：1架通信，2架探索 ==========
    print("\n" + "=" * 60)
    print("对照组1：1架通信，2架探索")
    print("=" * 60)
    
    obs_list_low, _ = env.reset()
    n_agents = len(env.uavs)
    terminated = truncated = False
    t = 0
    high_level_step = 0  # 高层步计数器
    
    # 固定策略：无人机0做通信，无人机1和2做探索
    fixed_options_1 = [1, 0, 0]  # 1=通信，0=探索
    
    while t < max_steps:
        high_level_step += 1  # 每个高层决策算一步
        # 固定策略：直接分配任务，但电量低于20%时选择补给
        options_this_step = fixed_options_1[:n_agents].copy()  # 确保数量匹配，使用copy避免修改原列表
        option_counts = np.zeros(n_options, dtype=np.int32)
        
        battery_threshold = 60  # 电量阈值：1000 * 0.2 = 200
        
        for i, option in enumerate(options_this_step):
            # 检查电量：如果电量低于20%，选择充电子任务（option=2）
            if env.uavs[i].e < battery_threshold:
                options_this_step[i] = 2  # 充电子任务
            else:
                options_this_step[i] = option  # 使用固定策略
            
            env.uavs[i].option = options_this_step[i]
            option_counts[options_this_step[i]] += 1
        
        env.option_counts = option_counts.copy()
        
        # 低层执行 K 步
        k = 0
        while k < K and t < max_steps:
            actions = []
            for i, obs_low in enumerate(obs_list_low):
                act = option_to_action(options_this_step[i], i, obs_low, env)
                actions.append(act)
            
            next_obs_list_low, rewards, terminated, truncated, infos = env.step(actions)
            # env.render()
            # 在每个低层时间步记录一次数据
            true_b = max(1, int(getattr(env, "true_boundary", 0)))
            known_b = int(getattr(env, "known_boundary", 0))
            explore_rate = float(known_b) / float(true_b) if true_b > 0 else 0.0
            coverage_rate = float(getattr(env, "coverage_rate", 0.0))
            
            all_data.append({
                "时间步": t + 1,  # 低层时间步从1开始
                "实验组": "对照组1(1通信2探索)",
                "探索率": explore_rate,
                "覆盖率": coverage_rate
            })
            
            at_station = any_uav_at_station(env)
            if at_station:
                obs_list_low = next_obs_list_low
                t += 1
                k += 1
                break
            
            obs_list_low = next_obs_list_low
            t += 1
            k += 1
            
            if terminated or truncated:
                # 对比实验不提前结束，仅跳出当前K循环
                break
        
        # 对比实验不提前结束，只有到达max_steps才结束
        terminated = False
        truncated = False
    
    print(f"对照组1完成，总步数: {t}")
    
    # ========== 对照组2：2架通信，1架探索 ==========
    print("\n" + "=" * 60)
    print("对照组2：2架通信，1架探索")
    print("=" * 60)
    
    obs_list_low, _ = env.reset()
    # 重置rescuers_had_target，避免因为救援人员没有目标而提前终止
    if hasattr(env, 'rescuers_had_target'):
        env.rescuers_had_target = False
    n_agents = len(env.uavs)
    terminated = truncated = False
    t = 0
    high_level_step = 0  # 高层步计数器
    
    # 固定策略：无人机0和1做通信，无人机2做探索
    fixed_options_2 = [1, 1, 0]  # 1=通信，0=探索
    
    while t < max_steps:
        high_level_step += 1  # 每个高层决策算一步
        # 固定策略：直接分配任务，但电量低于20%时选择补给
        options_this_step = fixed_options_2[:n_agents].copy()  # 确保数量匹配，使用copy避免修改原列表
        option_counts = np.zeros(n_options, dtype=np.int32)
        
        battery_threshold = 200  # 电量阈值：1000 * 0.2 = 200
        
        for i, option in enumerate(options_this_step):
            # 检查电量：如果电量低于20%，选择充电子任务（option=2）
            if env.uavs[i].e < battery_threshold:
                options_this_step[i] = 2  # 充电子任务
            else:
                options_this_step[i] = option  # 使用固定策略
            
            env.uavs[i].option = options_this_step[i]
            option_counts[options_this_step[i]] += 1
        
        env.option_counts = option_counts.copy()
        
        # 低层执行 K 步
        k = 0
        while k < K and t < max_steps:
            actions = []
            for i, obs_low in enumerate(obs_list_low):
                act = option_to_action(options_this_step[i], i, obs_low, env)
                actions.append(act)
            
            next_obs_list_low, rewards, terminated, truncated, infos = env.step(actions)
            # env.render()
            
            # 在每个低层时间步记录一次数据
            true_b = max(1, int(getattr(env, "true_boundary", 0)))
            known_b = int(getattr(env, "known_boundary", 0))
            explore_rate = float(known_b) / float(true_b) if true_b > 0 else 0.0
            coverage_rate = float(getattr(env, "coverage_rate", 0.0))
            
            all_data.append({
                "时间步": t + 1,  # 低层时间步从1开始
                "实验组": "对照组2(2通信1探索)",
                "探索率": explore_rate,
                "覆盖率": coverage_rate
            })
            
            at_station = any_uav_at_station(env)
            if at_station:
                obs_list_low = next_obs_list_low
                t += 1
                k += 1
                break
            
            obs_list_low = next_obs_list_low
            t += 1
            k += 1
            
            # 对于对照组2，如果是因为救援人员没有目标而终止，忽略这个终止条件，继续运行
            if terminated:
                # 检查是否是因为救援人员没有目标而终止
                all_rescuers_no_target = True
                for rescuer in getattr(env, 'rescuers', []):
                    if rescuer.target_pos is not None:
                        all_rescuers_no_target = False
                        break
                
                # 如果是因为救援人员没有目标而终止，且还没有达到最大步数，继续运行
                if all_rescuers_no_target and hasattr(env, 'rescuers_had_target') and env.rescuers_had_target and t < max_steps:
                    terminated = False  # 忽略这个终止条件
                    print(f"[对照组2] 忽略救援人员无目标的终止条件，继续运行 (t={t})")
            
            if terminated or truncated:
                # 对比实验不提前结束，仅跳出当前K循环
                break
        
        # 对比实验不提前结束，只有到达max_steps才结束
        terminated = False
        truncated = False
    
    print(f"对照组2完成，总步数: {t}")
    
    # ========== 保存数据到CSV ==========
    print("\n" + "=" * 60)
    print("保存数据到CSV")
    print("=" * 60)
    
    df = pd.DataFrame(all_data)
    
    # 重新组织数据格式：每个时间步一行，包含三个实验组的数据
    # 先按时间步分组
    time_steps = sorted(df["时间步"].unique())
    result_rows = []
    
    for t_step in time_steps:
        row_data = {"时间步": t_step}
        
        # 获取该时间步下各实验组的数据
        for group_name in ["强化学习", "对照组1(1通信2探索)", "对照组2(2通信1探索)"]:
            group_data = df[(df["时间步"] == t_step) & (df["实验组"] == group_name)]
            if len(group_data) > 0:
                row_data[f"{group_name}_探索率"] = group_data.iloc[0]["探索率"]
                row_data[f"{group_name}_覆盖率"] = group_data.iloc[0]["覆盖率"]
            else:
                row_data[f"{group_name}_探索率"] = None
                row_data[f"{group_name}_覆盖率"] = None
        
        result_rows.append(row_data)
    
    result_df = pd.DataFrame(result_rows)
    
    # 确保保存路径是CSV格式
    if not save_path.endswith('.csv'):
        save_path = save_path.replace('.xlsx', '.csv').replace('.xls', '.csv')
    
    # 保存为CSV
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"数据已保存到CSV: {save_path}")
    print(f"共记录 {len(result_df)} 个时间步的数据")
    
    # 打印统计信息
    print("\n统计信息:")
    print("-" * 60)
    for group_name in ["强化学习", "对照组1(1通信2探索)", "对照组2(2通信1探索)"]:
        group_data = df[df["实验组"] == group_name]
        if len(group_data) > 0:
            avg_explore = group_data["探索率"].mean()
            avg_coverage = group_data["覆盖率"].mean()
            final_explore = group_data["探索率"].iloc[-1] if len(group_data) > 0 else 0
            final_coverage = group_data["覆盖率"].iloc[-1] if len(group_data) > 0 else 0
            print(f"{group_name}:")
            print(f"  平均探索率: {avg_explore:.4f}")
            print(f"  平均覆盖率: {avg_coverage:.4f}")
            print(f"  最终探索率: {final_explore:.4f}")
            print(f"  最终覆盖率: {final_coverage:.4f}")
    
    return result_df


if __name__ == "__main__":
    """
    主函数：支持训练和测试两种模式
    
    训练模式：
    - 模式A（有拓扑）：use_topology_obs=True, use_topology_exploration=True
    - 模式B（无拓扑）：use_topology_obs=False, use_topology_exploration=False
    
    测试模式：
    - test_trained_model: 测试训练好的模型
    - comparison_experiment: 对比实验
    - coverage_explore_experiment: 覆盖率/探索率实验
    """
    
    # ========== 训练模式配置 ==========
    # 选择训练模式：'topology' (有拓扑) 或 'no_topology' (无拓扑)
    TRAIN_MODE = 'topology'  # 可选: 'topology' 或 'no_topology'
    
    # 训练参数
    NUM_EPISODES = 1501
    MAX_STEPS = 120
    N_OPTIONS = 3
    RESUME_FROM = None  # 从检查点恢复训练，例如: "hrl_dqn/Result/model/new_checkpoint_ep1000.pth"
    
    # 根据模式设置拓扑参数
    if TRAIN_MODE == 'topology':
        # 模式A：有拓扑（Proposed）
        USE_TOPOLOGY_OBS = True
        USE_TOPOLOGY_EXPLORATION = True
        print("=" * 60)
        print("训练模式：有拓扑（Proposed）")
        print("  - 使用拓扑观测（火头/侧翼/火尾长度）")
        print("  - 使用拓扑感知的探索调度")
        print("=" * 60)
    elif TRAIN_MODE == 'no_topology':
        # 模式B：无拓扑（Ablation）
        USE_TOPOLOGY_OBS = False
        USE_TOPOLOGY_EXPLORATION = False
        print("=" * 60)
        print("训练模式：无拓扑（Ablation）")
        print("  - 拓扑观测置零")
        print("  - 使用基础epsilon衰减")
        print("=" * 60)
    else:
        raise ValueError(f"未知的训练模式: {TRAIN_MODE}，请选择 'topology' 或 'no_topology'")

    # visualize_trained_model(
    #     model_path="hrl_dqn_improve/Result_v2/new_checkpoint_topology_ep1500.pth",
    #     max_steps=MAX_STEPS,
    #     n_options=N_OPTIONS,
    #     use_topology_obs=USE_TOPOLOGY_OBS,
    #     use_topology_exploration=USE_TOPOLOGY_EXPLORATION,
    #     pause_time=0  # 渲染暂停时间（秒），可调整动画速度
    # )
    # ========== 开始训练 ==========
    train_high_level(
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        n_options=N_OPTIONS,
        resume_from=RESUME_FROM,
        use_topology_obs=USE_TOPOLOGY_OBS,
        use_topology_exploration=USE_TOPOLOGY_EXPLORATION
    )
    
    # ========== 可视化模式（已注释，需要时取消注释） ==========
    # 加载训练好的模型并可视化运行一轮

    
    