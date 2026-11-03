# -*- coding: utf-8 -*-
from collections import deque
from matplotlib import pyplot as plt
import numpy as np
from typing import List, Dict, Any
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Env.低层环境 import GridEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from 探索任务 import (                                     # 启发式工具:contentReference[oaicite:4]{index=4}
    heuristic_action,               # 侦察：obs_low -> action_idx
    select_band_and_boundary,       # 预处理，写入 uav.B
    select_flank_by_ybands,
    select_tail_by_xband,
    step_attack_head_minimal,        # 火头灭火：uav/map/env -> action_idx
    step_attack_flank_minimal,
    step_attack_tail_minimal,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode="valid")
# class QNet(nn.Module):
#     def __init__(self, in_dim, n_options, hidden=256):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, hidden)
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.fc3 = nn.Linear(hidden, hidden)
#         self.out = nn.Linear(hidden, n_options)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         return self.out(x)
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
    def __init__(self, n_options=4, obs_dim=12, gamma=0.99, lr=1e-3,
                 eps_start=1.0, eps_end=0.05, eps_decay=2000,
                 buf_size=1000000, batch_size=128, target_tau=0.005, device=device):

        self.n_options = n_options
        self.gamma = gamma
        self.device = device
        self.n_step = 5                        # ← 你的目标里用了 γ**5

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
        return self.eps_s + frac * (self.eps_e - self.eps_s)
    
    import numpy as np

    def select_option_qifa(self, high_obs) -> int:
        """
        0=探索, 1=火头, 2=侧翼, 3=尾部
        索引：
            ammo       = obs[2]
            coverage   = obs[4]
            head_len   = obs[6]
            flank_len  = obs[7]
            tail_len   = obs[8]
            dist_head  = obs[-3]
            dist_flank = obs[-2]
            dist_tail  = obs[-1]
        """
        import numpy as np
        EPS = 1e-6
        NINF = -1e9

        # 读观测
        x_norm, y_norm = float(high_obs[0]), float(high_obs[1])
        ammo       = float(high_obs[2])
        coverage   = float(high_obs[4])
        head_len   = float(high_obs[6])
        flank_len  = float(high_obs[7])
        tail_len   = float(high_obs[8])
        dist_head  = max(0.0, float(high_obs[-3]))
        dist_flank = max(0.0, float(high_obs[-2]))
        dist_tail  = max(0.0, float(high_obs[-1]))

        # 低覆盖率：只探索
        if coverage < 0.85 - EPS:
            return 0

        # 存在性 gating（核心修复点）
        exist_h = head_len  > EPS
        exist_f = flank_len > EPS
        exist_t = tail_len  > EPS
        exist_cnt = int(exist_h) + int(exist_f) + int(exist_t)

        # 三类都没了 → 探索
        if exist_cnt == 0:
            return 0
        # 只有一类 → 直接选它（避免在近零分上抖动错选）
        if exist_cnt == 1:
            if exist_h: return 1
            if exist_f: return 2
            return 3

        # 超参数（保持你原来风格）
        h0, a_need_h = 4.0, 0.6
        f0, a_need_f = 5.0, 0.6
        t0, a_need_t = 6.0, 0.6

        d_head, d_flank, d_tail = 10.0, 12.0, 14.0
        w_ammo, w_dist = 0.5, 0.5

        FIRE_BIAS   = 0.15
        CLOSE_BONUS = 0.15
        CLOSE_D_H, CLOSE_D_F, CLOSE_D_T = 3.0, 3.0, 3.0

        sig = lambda L, a, c: 1.0 / (1.0 + np.exp(-a * (L - c)))
        tanh_prox = lambda d, s: 1.0 - np.tanh(max(0.0, d) / max(EPS, s))
        close_bonus = lambda d, D: (CLOSE_BONUS * (1.0 - np.tanh(d / max(EPS, D)))) if d <= D else 0.0

        # 需求度（按长度）
        need_h = sig(head_len,  a_need_h, h0) if exist_h else 0.0
        need_f = sig(flank_len, a_need_f, f0) if exist_f else 0.0
        need_t = sig(tail_len,  a_need_t, t0) if exist_t else 0.0

        # 能力（弹药 + 接近度）
        ammo_cl = np.clip(ammo, 0.0, 1.0)
        proximity_h = tanh_prox(dist_head,  d_head)  if exist_h else 0.0
        proximity_f = tanh_prox(dist_flank, d_flank) if exist_f else 0.0
        proximity_t = tanh_prox(dist_tail,  d_tail)  if exist_t else 0.0

        cap_h = w_ammo * ammo_cl + w_dist * proximity_h
        cap_f = w_ammo * ammo_cl + w_dist * proximity_f
        cap_t = w_ammo * ammo_cl + w_dist * proximity_t

        # 任务打分（对不存在的任务直接给 -∞，彻底禁用）
        score_h = (need_h * cap_h + FIRE_BIAS * need_h + close_bonus(dist_head,  CLOSE_D_H)) if exist_h else NINF
        score_f = (need_f * cap_f + FIRE_BIAS * need_f + close_bonus(dist_flank, CLOSE_D_F)) if exist_f else NINF
        score_t = (need_t * cap_t + FIRE_BIAS * need_t + close_bonus(dist_tail,  CLOSE_D_T)) if exist_t else NINF

        # 探索打分：整体需求低 + 距离远 时更倾向探索
        avg_need = (need_h + need_f + need_t) / max(1, exist_cnt)
        avg_far  = 1.0 - (proximity_h + proximity_f + proximity_t) / max(1, exist_cnt)
        score_e  = (1.0 - avg_need) * avg_far

        # 轻微确定性抖动（打散并列；不影响被屏蔽类）
        score_h += 0.02 * ((x_norm * 31.0 + y_norm * 17.0) % 1.0) if exist_h else 0.0
        score_f += 0.02 * ((x_norm * 23.0 + y_norm * 19.0) % 1.0) if exist_f else 0.0
        score_t += 0.02 * ((x_norm * 29.0 + y_norm * 11.0) % 1.0) if exist_t else 0.0
        score_e += 0.01 * ((x_norm * 13.0 + y_norm * 29.0) % 1.0)

        # 最终选择
        scores = [score_e, score_h, score_f, score_t]
        return int(np.argmax(scores))


    # def select_option_qifa(self, high_obs: np.ndarray) -> int:
    #     """
    #     选项编码: 0=探索, 1=火头, 2=侧翼, 3=尾部

    #     high_obs 索引约定：
    #         coverage = high_obs[4]
    #         head_len = high_obs[6]
    #         flank_len = high_obs[7]
    #         tail_len = high_obs[8]
    #         dist_head  = high_obs[-3]
    #         dist_flank = high_obs[-2]
    #         dist_tail  = high_obs[-1]
    #     """
    #     EPS = 1e-6

    #     coverage   = float(high_obs[4])
    #     head_len   = float(high_obs[6])
    #     flank_len  = float(high_obs[7])
    #     tail_len   = float(high_obs[8])
    #     dist_head  = float(high_obs[-3])
    #     dist_flank = float(high_obs[-2])
    #     dist_tail  = float(high_obs[-1])

    #     # 1️⃣ 覆盖率 < 0.9 → 探索
    #     if coverage < 0.90 - EPS:
    #         return 0

    #     # 2️⃣ 火头+侧翼均存在 → 按规则比较
    #     if head_len > EPS and flank_len > EPS:
    #         score_head  = head_len * (20-dist_head)
    #         score_flank = (flank_len * (20-dist_flank)) / 8.0
    #         return 1 if score_head > score_flank else 2

    #     # 3️⃣ 只有火头
    #     if head_len > EPS:
    #         return 1

    #     # 4️⃣ 没有火头，只有侧翼
    #     if flank_len > EPS:
    #         return 2

    #     # 5️⃣ 火头+侧翼都没了，但尾部还在
    #     if tail_len > EPS:
    #         return 3

    #     # 6️⃣ 全部没了 → 探索
    #     return 0

        

    def select_option_qifa1(self, high_obs: np.ndarray) -> int:
        """
        0=探索, 1=灭火
        high_obs: [pos(2), ammo(1), g..., dist2fire(1)]
                索引: ammo=2, coverage=4, head_len=6, dist2fire=-1
        规则：coverage<0.90 一定探索；head_len==0（无火）一定探索；
            其余情况用打分制，近火/弹药多/火大 → 更倾向灭火（但非一刀切）。
        """
        import numpy as np
        EPS = 1e-6

        # 取关键信息
        x_norm, y_norm = float(high_obs[0]), float(high_obs[1])  # 仅用于确定性抖动
        ammo       = float(high_obs[2])
        coverage   = float(high_obs[4])
        head_len   = float(high_obs[6])
        dist2fire  = float(high_obs[-1])

        # -------- 硬约束：低覆盖率 or 无火 → 必探索 --------
        if coverage < 0.90 - EPS:
            return 0
        if head_len <= 0 + EPS:
            return 0

        # -------- 以下仅在 coverage>=0.90 且 head_len>0 时执行 --------
        # 超参数（可按需要微调）
        h0, a_need = 4.0, 0.6      # 灭火“需求度”sigmoid中心与陡峭度
        d_scale    = 10.0          # 距离尺度
        w_ammo, w_dist = 0.5, 0.5  # 能力中弹药/距离的权重

        FIRE_BIAS      = 0.20      # 有火时整体偏向灭火（软增强）
        CLOSE_BONUS_D  = 3.0       # 认为“贴近火头”的距离阈值
        CLOSE_BONUS    = 0.20      # 贴近火头的额外加成

        BASE_FIRE_W = 1.0
        BASE_EXPL_W = 1.0

        # 需求度（火越大越需要灭）
        need_fire = 1.0 / (1.0 + np.exp(-a_need * (head_len - h0)))  # 0~1

        # 能力（越近火/弹药越多越强）
        proximity = 1.0 - np.tanh(max(0.0, dist2fire) / d_scale)     # 近→~1, 远→~0
        ammo_cl   = np.clip(ammo, 0.0, 1.0)
        capability = w_ammo * ammo_cl + w_dist * proximity           # 0~1

        # 基础打分
        score_fire = BASE_FIRE_W * (need_fire * capability)
        score_expl = BASE_EXPL_W * ((1.0 - need_fire) * (1.0 - proximity))

        # 有火偏置 & 贴近加成（只加在灭火侧）
        score_fire += FIRE_BIAS * need_fire
        if dist2fire <= CLOSE_BONUS_D:
            score_fire += CLOSE_BONUS * (1.0 - np.tanh(dist2fire / CLOSE_BONUS_D))

        # 轻微确定性抖动，打散并列
        score_fire += 0.02 * ((x_norm * 31.0 + y_norm * 17.0) % 1.0)
        score_expl += 0.01 * ((x_norm * 13.0 + y_norm * 29.0) % 1.0)

        # 最终选择
        gap = score_fire - score_expl

        return 1 if gap > 0.0 else 0



    def eps_rand(self, ep, start=0.3, end=0.05, decay_ep=2000):
        """
        随机探索概率：从 start 线性衰减到 end
        ep: 当前 episode
        """
        slope = (end - start) / decay_ep
        val = start + slope * ep
        return float(np.clip(val, end, start))


    def eps_heur(self, ep, start=0.7, end=0.1, decay_ep=10000):
        """
        启发式探索概率：从 start 线性衰减到 end
        ep: 当前 episode
        """
        slope = (end - start) / decay_ep
        val = start + slope * ep
        return float(np.clip(val, end, start))
    
    def select_option(self, obs_high, ep):
        q = self.online(torch.from_numpy(obs_high).float().unsqueeze(0).to(self.device))
        q = q.detach().cpu().numpy()[0]

        # 掩码：不存在的任务置为 -inf
        head_len, flank_len, tail_len = obs_high[6], obs_high[7], obs_high[8]
        mask = np.array([1, head_len>0, flank_len>0, tail_len>0], dtype=bool)
        q[~mask] = -1e9

        p_rand  = self.eps_rand(ep)
        p_heur  = self.eps_heur(ep)
        u = np.random.rand()

        if u < p_rand:
            valid_ids = np.where(mask)[0]
            return int(np.random.choice(valid_ids))
        elif u < p_rand + p_heur:
            return int(self.select_option_qifa(obs_high))
        else:
            return int(np.argmax(q))



    def push(self, tr):  # tr: dict with obs_high, option, Rk, next_obs_high, done
        self.replay.append(tr)

    def soft_update(self):
        # target ← τ*online + (1-τ)*target
        for tp, p in zip(self.target.parameters(), self.online.parameters()):
            tp.data.mul_(1-self.target_tau).add_(self.target_tau*p.data)

    def _mask_from_obs_tensor(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        obs_tensor: [B, obs_dim] (位于 self.device)
        返回: bool mask [B, n_options]，其中列 1/2/3 取决于 head/flank/tail 是否存在
        """
        B = obs_tensor.size(0)
        mask = torch.ones(B, self.n_options, dtype=torch.bool, device=self.device)
        head_len  = obs_tensor[:, 6]
        flank_len = obs_tensor[:, 7]
        tail_len  = obs_tensor[:, 8]
        mask[:, 1] = head_len  > 0.0
        mask[:, 2] = flank_len > 0.0
        mask[:, 3] = tail_len  > 0.0
        return mask

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        import random, numpy as np, torch
        batch = random.sample(self.replay, self.batch_size)

        obs  = torch.tensor(np.stack([b["obs_high"] for b in batch]), dtype=torch.float32, device=self.device)
        act  = torch.tensor([b["option"] for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        Rk   = torch.tensor([b["Rk"] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        nxt  = torch.tensor(np.stack([b["next_obs_high"] for b in batch]), dtype=torch.float32, device=self.device)
        done = torch.tensor([b["done"] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        # -------- Double DQN target with mask --------
        with torch.no_grad():
            # online 选动作（掩掉无效）
            q_next_online = self.online(nxt)                                  # [B,4]
            mask_next = self._mask_from_obs_tensor(nxt)                       # [B,4] bool
            q_next_online = q_next_online.masked_fill(~mask_next, -1e9)
            nxt_act = q_next_online.argmax(dim=1, keepdim=True)               # [B,1]

            # target 评估该动作
            q_next_target = self.target(nxt).gather(1, nxt_act)               # [B,1]

            y = Rk + (1.0 - done) * (self.gamma ** self.n_step) * q_next_target

        # 当前 Q
        q = self.online(obs).gather(1, act)                                   # [B,1]

        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.opt.step()
        self.soft_update()
        return loss.item()

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



def compute_global_features(env: GridEnv) -> np.ndarray:
    """按你的计划抽取宏观状态：风、火区面积/强度、侧翼近似（边界长度）"""
    M = env.grid["burning_state"]
    H, W = M.shape
    before_fire = np.count_nonzero(env.grid["burning_state"] > 0)
    #1.风速
    wind_speed = float(env.wind_speed) / 10

    #2.探索率
    rate = env.known_boundary / env.true_boundary

    # print(f"探索率为{rate},已知的边界为{known_cnt},真实的边界为{true_cnt}")

    #3.火势增长
    fire_now = np.count_nonzero(env.grid["burning_state"] > 0)
    # print(f"当前的过火格子为{fire_now},上一个时间的过火格子为{env.fire_cells_prev},插值为{fire_now - env.fire_cells_prev}")
    fire_diff = fire_now - env.fire_cells_prev
    env.fire_cells_prev = fire_now

    #2.各火区长度
    head_len = len(env.uavs[0].B) / 10
    flank_len = len(env.uavs[0].B_flank) / 10
    tail_len = len(env.uavs[0].B_tail) / 10
    #3.侧翼长

    return np.array([wind_speed,float(rate),fire_diff,head_len,flank_len,tail_len],dtype=np.float32)

# ============== 3) 低层 -> 高层 观测重构 ==============
def reconstruct_high_obs(
    obs_low: np.ndarray,
    env: GridEnv,
    count: np.ndarray = None,   # 目前不再用于网络输入，保留参数以兼容调用
) -> np.ndarray:
    import numpy as np

    # 个体特征
    pos  = obs_low[15:17]      # 归一化位置 [x/50, y/50]
    ammo = obs_low[17:18]      # 归一化弹药

    # 全局特征
    g = compute_global_features(env)  # e.g. [wind, coverage, fire_diff, head_len, ...]

    # === 最近火头距离（用 pos 反推栅格坐标，取整）===
    H, W = env.grid["burning_state"].shape
    uav_x = int(round(float(pos[0]) * H))
    uav_y = int(round(float(pos[1]) * W))

    # 火头集合：优先用 uav.B；为空则退化为 burning_state==1 的格子
    if hasattr(env.uavs[0], "B") and isinstance(env.uavs[0].B, (list, tuple)) and len(env.uavs[0].B) > 0:
        fire_cells = list(env.uavs[0].B)
    else:
        xs, ys = np.where(np.isclose(env.grid["burning_state"], 1.0))
        fire_cells = list(zip(xs.tolist(), ys.tolist()))
    #侧翼集合
    if hasattr(env.uavs[0], "B_flank") and isinstance(env.uavs[0].B_flank, (list, tuple)) and len(env.uavs[0].B_flank) > 0:
        fire_cells_flank = list(env.uavs[0].B_flank)
    else:
        xs, ys = np.where(np.isclose(env.grid["burning_state"], 1.0))
        fire_cells_flank = list(zip(xs.tolist(), ys.tolist()))
    #尾部集合
    if hasattr(env.uavs[0], "B_tail") and isinstance(env.uavs[0].B_tail, (list, tuple)) and len(env.uavs[0].B_tail) > 0:
        fire_cells_tail = list(env.uavs[0].B_tail)
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
    # 拼接为高层输入（这里不再拼 count，避免顺序依赖；若你仍想保留，可以在最后再拼上）
    high_obs = np.concatenate([
        pos.astype(np.float32),
        ammo.astype(np.float32),
        np.asarray(g, dtype=np.float32),
        dis2fire/10,
        dis2fire_flank/10,
        dis2fire_tail/10,
    ], axis=0).astype(np.float32)

    return high_obs

# ============== 4) option -> 启发式动作 ==============
def option_to_action(option: int, agent_index: int, obs_low: np.ndarray, env: GridEnv) -> int:
    """
    把高层 option（子任务）映射为具体低层离散动作（0..8）
    0: 侦察（explore）
    1: 火头灭火（attack head）
    后续你可扩展 2: 阻火带、3: 侧翼、4: 清理余火（见计划）:contentReference[oaicite:7]{index=7}
    """
    if option == 0:
        uav = env.uavs[agent_index]
        select_band_and_boundary(uav)
        select_flank_by_ybands(uav,1)
        select_tail_by_xband(uav,1)
        return heuristic_action(obs_low)  # 侦察:contentReference[oaicite:8]{index=8}
    elif option == 1:
        uav = env.uavs[agent_index]
        select_band_and_boundary(uav)     # 计算边界候选，写入 uav.B:contentReference[oaicite:9]{index=9}
        select_flank_by_ybands(uav,1)
        select_tail_by_xband(uav,1)
        return step_attack_head_minimal(uav, uav.map_info, env)  # 产出一步动作:contentReference[oaicite:10]{index=10}
    elif option ==2:
        uav = env.uavs[agent_index]
        select_band_and_boundary(uav)     # 计算边界候选，写入 uav.B:contentReference[oaicite:9]{index=9}
        select_flank_by_ybands(uav,1)
        select_tail_by_xband(uav,1)
        return step_attack_flank_minimal(uav, uav.map_info, env)  # 产出一步动作:contentReference[oaicite:10]{index=10}
    elif option ==3:
        uav = env.uavs[agent_index]
        select_band_and_boundary(uav)     # 计算边界候选，写入 uav.B:contentReference[oaicite:9]{index=9}
        select_flank_by_ybands(uav,1)
        select_tail_by_xband(uav,1)
        return step_attack_tail_minimal(uav, uav.map_info, env)  # 产出一步动作:contentReference[oaicite:10]{index=10}
    else:
        return 8  # 未定义则“原地”
    

def prefill_replay_with_heuristics(policy, env, K=5, target_prefill=5000, max_steps=200):
    """
    用启发式（任务切换器/规则）生成高层 SMDP transition，填充到 policy.replay。
    关键点：与训练循环一致的时序——K步为主；若火头出现/消失，在K内提前重选（提前截断本次聚合）。
    transition 结构与训练一致：obs_high / option / Rk / next_obs_high / done
    """
    import numpy as np

    # —— 与训练一致的事件触发参数（可按需改）—

    def _head_len(_env):
        if len(_env.uavs) > 0 and hasattr(_env.uavs[0], "B") and isinstance(_env.uavs[0].B, (list, tuple)):
            return len(_env.uavs[0].B)
        return 0
    def _head_flank(_env):
        if len(_env.uavs) > 0 and hasattr(_env.uavs[0], "B_flank") and isinstance(_env.uavs[0].B_flank, (list, tuple)):
            return len(_env.uavs[0].B_flank)
    def _head_tail(_env):
        if len(_env.uavs) > 0 and hasattr(_env.uavs[0], "B_tail") and isinstance(_env.uavs[0].B_tail, (list, tuple)):
            return len(_env.uavs[0].B_tail)
        return 0

    added = 0
    n_options = getattr(env, "n_options", 4)

    while len(policy.replay) < target_prefill:
        obs_list_low, _ = env.reset()
        n_agents = len(env.uavs)
        t = 0

        # 每集的事件快照
        prev_head_len = _head_len(env)
        prev_flank_len = _head_flank(env)
        prev_tail_len = _head_tail(env)
        prev_cov = 0

        while t < max_steps and len(policy.replay) < target_prefill:
            # —— 1) 本轮高层决策（与训练一致：逐个决策+计数，占比来自“当前已选”）——
            options_this_step, high_obs_list = [], []
            option_counts = np.zeros(n_options, dtype=np.int32)

            for i, obs_low in enumerate(obs_list_low):
                # high_obs 里最后是“占比”，所以传 counts / n_agents
                high_obs = reconstruct_high_obs(obs_low=obs_low, env=env, count=option_counts)
                # 启发式“专家”：用你的 qifa（保持注入的“启发式”属性）
                # option = policy.select_option_qifa(high_obs, 2)
                option = policy.select_option_qifa(high_obs)
                env.uavs[i].option = option
                env.uavs[i].count_now = option_counts.copy()
                # print(f"无人机{i+1}的count_now是{env.uavs[i].count_now}")
                option_counts[option] += 1


                options_this_step.append(option)
                high_obs_list.append(high_obs)

            env.option_counts = option_counts

            # print(f"无人机1的count_now是{env.uavs[0].count_now}")
            # —— 2) 低层执行 K 步，遇到“火头出现/消失”则提前截断，形成一次高层聚合 —— 
            Rk = np.zeros(n_agents, dtype=np.float32)
            disc = 1.0
            k = 0

            # 记录“是否提前重选”（这里只用于截断本次聚合，下一轮会重新决策）

            while k < K and t < max_steps:
                # 记录 memory 起点（你原本的过滤逻辑用得到）
                mem_start = [int(obs[0]) for obs in obs_list_low]

                # 固定 option 生成动作
                actions = [
                    option_to_action(options_this_step[i], i, obs_low, env)
                    for i, obs_low in enumerate(obs_list_low)
                ]
                next_obs_list_low, rewards, terminated, truncated, infos = env.step(actions)

                # env.render()  # 可选

                # === 先做事件检测，再决定要不要计入本步奖励  ★ ===
                curr_head_len = _head_len(env)
                curr_flank_len = _head_flank(env)
                curr_tail_len = _head_tail(env)


                flip = (prev_head_len == 0 and curr_head_len > 0) or (prev_head_len > 0 and curr_head_len == 0)
                filp_flank = (prev_flank_len == 0 and curr_flank_len > 0) or (prev_flank_len > 0 and curr_flank_len == 0)
                filp_tail = (prev_tail_len == 0 and curr_tail_len > 0) or (prev_tail_len > 0 and curr_tail_len == 0)

                curr_cov = float(env.known_boundary) / max(1, float(env.true_boundary))
                flip_cov = (prev_cov < 0.9 and curr_cov >= 0.9) or (prev_cov >= 0.9 and curr_cov < 0.9)

                if  flip or flip_cov or filp_flank or filp_tail:
                    # 触发提前重选：本步不累计奖励、不更新折扣  ★
                    prev_head_len = curr_head_len
                    prev_flank_len = curr_flank_len
                    prev_tail_len = curr_tail_len
                    prev_cov = curr_cov

                    obs_list_low = next_obs_list_low
                    t += 1
                    k += 1

                    break

                # —— 未触发事件，才累计本步奖励与折扣  ★ ——
                Rk   += disc * np.array(rewards, dtype=np.float32)
                disc *= policy.gamma

                prev_head_len = curr_head_len
                prev_flank_len = curr_flank_len
                prev_tail_len = curr_tail_len
                prev_cov = curr_cov

                obs_list_low = next_obs_list_low
                t += 1
                k += 1

                if terminated or truncated:
                    break

            # —— 3) 形成高层 transition，push 到 replay（结构与训练一致） —— 
            next_high_obs_list = [
                reconstruct_high_obs(obs_low=obs_list_low[i], env=env, count=np.zeros(n_options, dtype=np.int32))
                for i in range(n_agents)
            ]
            done_flag = bool(terminated or truncated) if ('terminated' in locals()) else False

            # 只保留“有学习意义的段”（你原来的过滤：灭火 或 探索但 mem_start==1）
            for i in range(n_agents):
                opt = options_this_step[i]
                eligible = (opt in [1,2,3]) or (opt == 0 and mem_start[i] == 1)
                if not eligible:
                    # print(f"无人机{i}正在赶路")
                    continue

                tr = {
                    "obs_high":      high_obs_list[i],
                    "option":        int(options_this_step[i]),
                    "Rk":            float(Rk[i]),
                    "next_obs_high": next_high_obs_list[i],
                    "done":          done_flag,
                }
                # print(tr)
                policy.push(tr)
                added += 1

            if done_flag:
                break
            # print(f"奖励为{Rk}")
            # print("--------------------------------------------------\n")

    print(f"[Prefill] 预填启发式经验完成，共加入 {added} 条样本（当前 replay={len(policy.replay)}）")

import torch
import numpy as np

def test_trained_model(model_path="Result/model/checkpoint_ep1000.pth",
                       n_episodes=5, max_steps=200):
    import numpy as np
    import torch

    env = GridEnv()
    policy = HighLevelPolicy(n_options=2)  # 与训练一致

    # ==== 加载模型 ====
    checkpoint = torch.load(model_path, map_location=policy.device)
    policy.online.load_state_dict(checkpoint["online_state_dict"])
    policy.target.load_state_dict(checkpoint["target_state_dict"])
    policy.opt.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"[Test] 模型已加载 {model_path}")


    def _head_len():
        if len(env.uavs) > 0 and hasattr(env.uavs[0], "B") and isinstance(env.uavs[0].B, (list, tuple)):
            return len(env.uavs[0].B)
        return 0

    returns = []
    K = 5  # 和训练一致

    for ep in range(n_episodes):
        obs_list_low, _ = env.reset()
        n_agents = len(env.uavs)

        # 事件状态快照（与训练一致）
        prev_head_len = _head_len()
        prev_cov = float(env.known_boundary) / max(1.0, float(env.true_boundary))

        ep_ret = np.zeros(n_agents, dtype=np.float32)
        terminated = truncated = False  # 防止未定义
        t = 0

        while t < max_steps:
            # ===== 1) 高层决策（贪心） =====
            options_this_step = []
            high_obs_list = []
            option_counts = np.zeros(2, dtype=np.int32)

            for i, obs_low in enumerate(obs_list_low):
                high_obs = reconstruct_high_obs(obs_low=obs_low, env=env, count=option_counts)
                with torch.no_grad():
                    q = policy.online(
                        torch.tensor(high_obs, dtype=torch.float32, device=policy.device).unsqueeze(0)
                    )
                    option = int(q.argmax(1).item())
                env.uavs[i].option = option
                env.uavs[i].count_now = option_counts.copy()  # 与训练保持一致的快照
                option_counts[option] += 1

                options_this_step.append(option)
                high_obs_list.append(high_obs)

            env.option_counts = option_counts

            # ===== 2) 低层执行 K 步（或事件触发提前重选） =====
            k = 0
            while k < K and t < max_steps:
                # 固定 option 生成动作
                actions = [
                    option_to_action(options_this_step[i], i, obs_low, env)
                    for i, obs_low in enumerate(obs_list_low)
                ]
                next_obs_list_low, rewards, terminated, truncated, infos = env.step(actions)

                # === 事件检测先于奖励累计（与训练、prefill一致） ===
                curr_head_len = _head_len()
                flip_head = ((prev_head_len == 0 and curr_head_len > 0) or
                             (prev_head_len > 0 and curr_head_len == 0))

                curr_cov = float(env.known_boundary) / max(1.0, float(env.true_boundary))
                flip_cov = ((prev_cov < 0.9 and curr_cov >= 0.9) or
                            (prev_cov >= 0.9 and curr_cov < 0.9))

                if  flip_head or flip_cov:
                    prev_head_len = curr_head_len
                    prev_cov = curr_cov

                    obs_list_low = next_obs_list_low
                    t += 1
                    k += 1

                    # if flip_head:
                    #     print("[Test] ---- 火头发生变动！提前重选 ----")
                    # elif flip_cov:
                    #     print("[Test] ---- 覆盖率跨过0.9阈值！提前重选 ----")
                    break  # 立刻跳出本轮K，外层马上重新做一次高层决策

                # —— 未触发事件：才累计本步奖励
                ep_ret += np.array(rewards, dtype=np.float32)
                print(f"这一轮的选择是{options_this_step}")
                print(f"这一轮的奖励是{rewards}")
                print("")
                prev_head_len = curr_head_len
                prev_cov = curr_cov


                # 推进
                obs_list_low = next_obs_list_low
                t += 1
                k += 1

                # 可选：渲染
                env.render()

                if terminated or truncated:
                    break

            if terminated or truncated:
                break  # 结束 episode

        mean_ret = ep_ret.mean()
        returns.append(mean_ret)
        print(f"[Test EP {ep+1}] mean return/agent = {mean_ret:.3f}, steps = {t}")

    print(f"[Test Summary] 平均回报 = {np.mean(returns):.3f} ± {np.std(returns):.3f}")


# ============== 5) 高层训练主循环骨架（无网络，仅数据流） ==============
def train_high_level(
    num_episodes: int = 10000,
    max_steps: int = 200,
    n_options: int = 4
):
    K = 5  # 每次高层决策持续的低层步数
    env = GridEnv()
    policy = HighLevelPolicy(n_options=n_options)
    episode_rewards = []
    losses = []

    prefill_replay_with_heuristics(policy, env, K=K, target_prefill=5000, max_steps=max_steps)


    def _head_len():
        if len(env.uavs) > 0 and hasattr(env.uavs[0], "B") and isinstance(env.uavs[0].B, (list, tuple)):
            return len(env.uavs[0].B)
        return 0
    def _head_flank(_env):
        if len(_env.uavs) > 0 and hasattr(_env.uavs[0], "B_flank") and isinstance(_env.uavs[0].B_flank, (list, tuple)):
            return len(_env.uavs[0].B_flank)
    def _head_tail(_env):
        if len(_env.uavs) > 0 and hasattr(_env.uavs[0], "B_tail") and isinstance(_env.uavs[0].B_tail, (list, tuple)):
            return len(_env.uavs[0].B_tail)
        return 0
    
    for ep in range(num_episodes):
        obs_list_low, _ = env.reset()
        n_agents = len(env.uavs)

        # 每集重置事件状态（与 prefill 对齐：prev_head_len/prev_cov）
        prev_head_len = _head_len()
        prev_flank_len = _head_flank(env)
        prev_tail_len = _head_tail(env)

        prev_cov = float(env.known_boundary) / max(1.0, float(env.true_boundary))

        ep_ret = np.zeros(n_agents, dtype=np.float32)

        t = 0
        while t < max_steps:
            # ===== 1) 选一次高层 option（基于当前高层观测） =====
            options_this_step: List[int] = []
            high_obs_list: List[np.ndarray] = []

            option_counts = np.zeros(n_options, dtype=np.int32)

            for i, obs_low in enumerate(obs_list_low):
                # 与 prefill 一致：把“当前已选计数快照”灌入 high_obs（顺序依赖接受）
                high_obs = reconstruct_high_obs(obs_low=obs_low, env=env, count=option_counts)
                option = policy.select_option(high_obs, ep)
                env.uavs[i].option = option
                env.uavs[i].count_now = option_counts.copy()
                option_counts[option] += 1
                options_this_step.append(option)
                high_obs_list.append(high_obs)

            env.option_counts = option_counts

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
                curr_head_len = _head_len()
                curr_flank_len = _head_flank(env)
                curr_tail_len = _head_tail(env)

                flip_head = ((prev_head_len == 0 and curr_head_len > 0) or
                             (prev_head_len > 0 and curr_head_len == 0))
                filp_flank = (prev_flank_len == 0 and curr_flank_len > 0) or (prev_flank_len > 0 and curr_flank_len == 0)
                filp_tail = (prev_tail_len == 0 and curr_tail_len > 0) or (prev_tail_len > 0 and curr_tail_len == 0)

                curr_cov = float(env.known_boundary) / max(1.0, float(env.true_boundary))
                flip_cov = ((prev_cov < 0.9 and curr_cov >= 0.9) or
                            (prev_cov >= 0.9 and curr_cov < 0.9))
            
                if flip_head or flip_cov or filp_flank or filp_tail:
                    # 触发提前重选：本步不累计奖励、不推进折扣
                    prev_head_len = curr_head_len
                    prev_flank_len = curr_flank_len
                    prev_tail_len = curr_tail_len
                    prev_cov = curr_cov
                    obs_list_low = next_obs_list_low
                    t += 1
                    k += 1

                    break  # 立刻跳出 K 循环，回到外层进行新一轮高层决策

                # —— 未触发事件：才累计本步奖励与折扣
                reward_sum += disc * np.array(rewards, dtype=np.float32)
                disc *= gamma

                ep_ret += np.array(rewards, dtype=np.float32)

                prev_head_len = curr_head_len
                prev_flank_len = curr_flank_len
                prev_tail_len = curr_tail_len
                prev_cov = curr_cov
                obs_list_low = next_obs_list_low
                t += 1
                k += 1

                if terminated or truncated:
                    break


            # ===== 3) 形成一次“高层 transition”（K步聚合）=====
            next_high_obs_list = [
                reconstruct_high_obs(obs_low=obs_list_low[i], env=env, count=np.zeros(n_options, dtype=np.int32))
                for i in range(n_agents)
            ]
            done_flag = bool(terminated or truncated) if ('terminated' in locals()) else False

            for i in range(n_agents):
                opt = options_this_step[i]
                eligible = (opt in [1,2,3]) or (opt == 0 and mem_start[i] == 1)
                if not eligible:
                    # print(f"无人机{i}正在赶路")
                    continue
                transition = {
                    "obs_high":      high_obs_list[i],
                    "option":        options_this_step[i],
                    "Rk":            float(reward_sum[i]),
                    "next_obs_high": next_high_obs_list[i],
                    "done":          done_flag,
                }
                policy.push(transition)
                # print(transition["Rk"])
            loss = policy.update()
            if loss is not None:
                losses.append(loss)

            if done_flag:
                break  # 结束 episode
            # print("---------------------------------")

        # 一集结束
        total_reward = ep_ret.mean()
        episode_rewards.append(total_reward)
        print(f"[EP {ep:03d}] mean return/agent = {total_reward:.3f}, steps = {t}, 随机数为{policy.epsilon(ep)}")



        if (ep+1)%50 == 0 :
            smoothed_reward = moving_average(episode_rewards, window_size=50)
            plt.figure()
            plt.plot(episode_rewards, alpha=0.3, label="Raw Reward")
            plt.plot(smoothed_reward, color='orange', label="Smoothed Reward")
            plt.xlabel("Episode")
            plt.ylabel("Mean Return / Agent")
            plt.title("Training Reward Curve")
            plt.savefig("hrl_8.24/Result/pic/reward_curve.png")

            plt.figure()
            plt.plot(losses)
            plt.xlabel("Update Step")
            plt.ylabel("Q Loss")
            plt.title("Q-Loss Curve")
            plt.savefig("hrl_8.24/Result/pic/q_loss_curve.png")

        if (ep + 1) % 1000 == 0:
            save_path = f"hrl_8.24/Result/model/checkpoint_ep{ep+1}.pth"
            torch.save({
                "online_state_dict": policy.online.state_dict(),
                "target_state_dict": policy.target.state_dict(),
                "optimizer_state_dict": policy.opt.state_dict(),
                "replay_size": len(policy.replay)
            }, save_path)
            print(f"[Checkpoint] 模型已保存到 {save_path}")
if __name__ == "__main__":
    # test_trained_model("Result/model/checkpoint_ep2000.pth", n_episodes=5, max_steps=150)
    train_high_level(
        num_episodes=10000,
        max_steps=200,
        n_options=4
    )
