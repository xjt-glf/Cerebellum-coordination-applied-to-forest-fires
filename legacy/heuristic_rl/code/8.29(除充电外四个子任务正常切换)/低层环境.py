#10x10任务区域，固定出生点，随机一个火点，不扩散，自动投弹

import math
import gymnasium as gym  # ✅ 使用 gymnasium
from gymnasium import spaces  # ✅ 确保 spaces 也从 gymnasium 导入
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
directions_8 = [(1, 0), (1, 1), (0, 1),
              (-1, 1), (-1, 0),
              (-1, -1), (0, -1), (1, -1)]
directions_9 = [(1, 0), (1, 1), (0, 1),
              (-1, 1), (-1, 0),
              (-1, -1), (0, -1), (1, -1),(0,0)]
action_set = [
        np.array([1,0]), np.array([1,1]), np.array([0,1]),
        np.array([-1,1]), np.array([-1,0]), np.array([-1,-1]),
        np.array([0,-1]), np.array([1,-1]), np.array([0,0])
    ]
class UAV:
    def __init__(self, uav_id, x, y, e, d):
        self.uav_id = uav_id  # 无人机 ID
        self.x = x  # 当前位置
        self.y = y
        self.e = e  # 剩余电量
        self.d = d  # 载弹量
        self.received_uav_info = []

        self.map_info = np.zeros((50, 50))
        self.current_target = [0,0]

        self.memory_flag = False
        self.B=[]
        self.BP=[]

        self.B_flank = []
        self.BP_flank = []

        self.B_tail = []
        self.BP_tail = []

        self.option = 0
        self.count_now = np.zeros(4)

        self.prev_option = None          # 上一步的高层任务（0/1/2/3）
        self.option_hold = 0             # 连续保持同一任务的步数
        self.dist_prev = {"head":None,   # 上一时刻到各类集合最近距离（用于进展奖励）
                        "flank":None,
                        "tail":None}
        
class GridEnv(gym.Env):
    """
    自定义的网格环境，模拟无人机群搜索救火任务。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, render_mode: str = "human"):
        super(GridEnv, self).__init__()

        self.charging_stations = [(0, 15), (15, 0), (15, 29), (29, 15)]

        # 任务区域 200x200 网格 一格100m×100m
        self.grid_size = (50, 50)

        # 环境参数：温度、湿度、风速、风向
        self.temperature = 30.0  # 温度（摄氏度）
        self.humidity = 20  # 湿度（%）
        self.wind_level = 6  # 风级（级）
        self.wind_speed = 12  # 风速（m/s）
        self.wind_direction = 0.0  # 风向，0° 代表水平向右
        self.R0 = 0.029 * self.temperature + 0.047 * self.wind_level - 0.009 * self.humidity + 0.304  # 初始蔓延速度
        self.ks = 2  # 枯枝落叶的可燃物系数
        self.kw = math.e ** (0.1783 * self.wind_speed)

        # 每个网格的属性
        self.grid = np.zeros(self.grid_size, dtype=[
            ('flammability', np.float32),  # 可燃物系数
            ('height', np.float32),  # 高度
            ('visibility', np.float32),  # 可见度
            ('burning_state', np.float32),  # 燃烧状态 0~1
            ('spread_speed', np.float32, (8,))  # 8个方向的扩散速度
        ])

        self.burn_time_counter = np.zeros(self.grid_size, dtype=np.int32)

        # 创建无人机群
        self._initialize_uavs()

        # 初始化网格属性
        self._initialize_grid()

        # 定义动作空间
        self.action_space = spaces.Discrete(9)

        # 其他初始化
        self.render_mode = render_mode

        self.fire_update_counter = 0
        #通信距离
        self.communication_radius = 6
        #高层记录K步前的火灾点
        self.fire_cells_prev = 0
        #已知的边界
        self.known_boundary = 0
        #上一个时间步已知的边界
        self.know_boundary_pre = 0
        #真实的边界
        self.true_boundary = 1
        #上一个时间步灭掉的火点数
        self.extinguish_pre = 0
        #上一个时间步的探索率
        self.coverage_pre = 0.0
        #已选选项数
        self.option_counts = 4
        #是否到达覆盖率
        self._has_reached_th_on = False

        self.head_len_pre  = 0    # 上一时刻的三类长度（切换惩罚中的“是否集合已空”判断）
        self.flank_len_pre = 0
        self.tail_len_pre  = 0
    def reset(self):
        self._initialize_grid()
        self.fire_update_counter = 0
        self._initialize_uavs()

        for uav in self.uavs:
            # 用全局火图同尺寸
            H, W = self.grid["burning_state"].shape
            uav.map_info = np.full((H, W), -1.0, dtype=np.float32)
            # uav.map_info[25,25] = 1
        # 收集所有无人机观测
        observations = []
        all_positions = {u.uav_id: (u.x, u.y) for u in self.uavs}
        # print(all_positions)
        for uav in self.uavs:
            obs = self.get_observation(
                uav=uav,
                burning_state=self.grid["burning_state"],
                q_fire=(25,25),
                all_uav_positions=all_positions
            )
            observations.append(obs)

        self.fire_cells_prev = int(np.count_nonzero(self.grid["burning_state"] > 0))

        return observations, {}


    def _initialize_grid(self):
        """初始化网格的四个参数"""
        self.grid['flammability'] = np.random.uniform(0.2, 1.0, self.grid_size)



        try:
            height_df = pd.read_csv("hrl_8.24/terrain_height.csv", header=0)
            height_array = height_df.values.astype(np.float32)
            if height_array.shape != self.grid_size:
                raise ValueError(f"高度图尺寸不匹配，应为 {self.grid_size}，实际为 {height_array.shape}")
            self.grid['height'] = height_array
            # print("✅ 成功加载 terrain_height.csv 作为地形高度图")
        except Exception as e:
            print(f"❌ 加载 terrain_height.csv 失败，使用默认随机高度: {e}")
            self.grid['height'] = np.random.uniform(20, 100, self.grid_size)




        self.grid['visibility'] = np.random.uniform(0.5, 1.0, self.grid_size)
        self.grid['burning_state'] = 0  # 初始未燃烧

        self.burn_time_counter[:, :] = 0
        
        # 计算 8 个方向的扩散速度
        self._compute_spread_speed()

        # fire_position = [(25,25),(25,24),(24,24),(24,25),(24,26),(25,26),(26,26),(26,25),(26,24)]
        fire_position = [(i, j) for i in range(20, 30) for j in range(20, 30)]

        for x, y in fire_position:
            self.grid['burning_state'][x, y] = 1.0
        # print(f"点燃格子{fire_position}")
        self.update_burning_state(time_step=1)
        
    def _initialize_uavs(self):
        self.num_uavs = 3
        self.uavs = []

        for uav_id in range(1, self.num_uavs+1):
            # 随机选一个健康格子
            while True:
                rand_x = np.random.randint(3, self.grid_size[0]-3)
                rand_y = np.random.randint(3, self.grid_size[1]-3)
                if self.grid["burning_state"][rand_x, rand_y]==0:
                    break

            # 创建无人机
            uav = UAV(uav_id, rand_x, rand_y, e=500, d=10)
            self.uavs.append(uav)

    def check_memory(self,uav, burning_state, local_radius=1):
        """
        uav_x, uav_y: 无人机坐标
        burning_state: 2D矩阵，0=健康,1=燃烧,2=烧尽
        local_radius: 视野半径 (默认为3x3)
        """
        x_min = max(0, uav.x - local_radius)
        x_max = min(burning_state.shape[0], uav.x + local_radius + 1)
        y_min = max(0, uav.y - local_radius)
        y_max = min(burning_state.shape[1], uav.y + local_radius + 1)

        local_patch = burning_state[x_min:x_max, y_min:y_max]

        if np.any((local_patch == 1) | (local_patch == 2)):
            uav.memory_flag = True

            
    def compute_rotation_vector(self,uav_x, uav_y, q_fire):
        dx = uav_x - q_fire[0]
        dy = uav_y - q_fire[1]
        return np.array([dy, -dx], dtype=np.float32)
    
    def compute_nearest_agent_vector(self,uav_x, uav_y, all_uav_positions, self_id):
        min_dist = float('inf')
        nearest_vec = np.zeros(2)
        nearest_id = None

        for uid, (x, y) in all_uav_positions.items():
            if uid == self_id:
                continue
            diff = np.array([uav_x - x, uav_y - y], dtype=np.float32)
            dist = np.linalg.norm(diff)
            if dist < min_dist:
                min_dist = dist
                nearest_vec = diff
                nearest_id = uid

        return nearest_id, nearest_vec, min_dist
    
    def compute_avoidance_flag(self, self_id, nearest_id, nearest_dist, distance_threshold=1.0):
        if nearest_id is None:
            return False
        # if nearest_dist > distance_threshold:
        #     return False
        return self_id > nearest_id

    def get_local_fire_image(self, uav_x, uav_y, burning_state, local_radius=1):
        """
        返回局部火情，从左上到右下，flatten后顺序：123456789
        """
        x_min = max(0, uav_x - local_radius)
        x_max = min(burning_state.shape[0], uav_x + local_radius + 1)
        y_min = max(0, uav_y - local_radius)
        y_max = min(burning_state.shape[1], uav_y + local_radius + 1)

        local_patch = burning_state[x_min:x_max, y_min:y_max]
        # print(f"原始观测",local_patch)
        return local_patch.flatten(order="C").astype(np.float32)



    def get_observation(self, uav, burning_state, q_fire, all_uav_positions):
        """
        uav: 一个无人机对象（假设有id, x, y属性）
        burning_state: 火情矩阵
        q_fire: 火中心 [x,y]
        all_uav_positions: {id: (x,y)}
        """
        # 1. Memory
        self.check_memory(uav, burning_state,1)
        memory_value = np.array([1.0 if uav.memory_flag else 0.0], dtype=np.float32)

        # 2. Local fire image
        local_image = self.get_local_fire_image(uav.x, uav.y, burning_state)
        # print(f"拉平后{local_image}")
        # 3. Rotation vector
        rotation_vec = self.compute_rotation_vector(uav.x, uav.y, q_fire)
        norm_rot = np.linalg.norm(rotation_vec)
        if norm_rot > 0:
            rotation_vec = rotation_vec / norm_rot
        else:
            rotation_vec = np.zeros(2, dtype=np.float32)

        if len(all_uav_positions)<=1:
            nearest_id = None
            nearest_vec = np.zeros(2,dtype=np.float32)
            avoidance_flag = False
            avoidance_value = np.array([1.0 if avoidance_flag else 0.0], dtype=np.float32)
        else:
            # 4. Nearest agent vector
            nearest_id, nearest_vec,min_dist = self.compute_nearest_agent_vector(
                uav.x, uav.y, all_uav_positions, uav.uav_id
            )
            # norm_nearest = np.linalg.norm(nearest_vec)
            # if norm_nearest > 0:
            #     nearest_vec = nearest_vec / norm_nearest
            # else:
            #     nearest_vec = np.zeros(2, dtype=np.float32)

            # 5. Avoidance flag
            avoidance_flag = self.compute_avoidance_flag(uav.uav_id, nearest_id,min_dist)
            avoidance_value = np.array([1.0 if avoidance_flag else 0.0], dtype=np.float32)

        # 6.自身位置
        x_norm = uav.x /50.0
        y_norm = uav.y  /50.0
        pos_value = np.array([x_norm, y_norm], dtype=np.float32)
        # 拼接所有特征

        ammo_value = np.array([uav.d / 10.0], dtype=np.float32)

        obs = np.concatenate([
            memory_value,       # 1维
            local_image,        # 9维
            rotation_vec,       # 2维 (单位化)
            avoidance_value,    # 1维
            nearest_vec,         # 2维 (单位化)
            pos_value,           #2
            ammo_value           #1
        ])

        return obs



    from math import sqrt

    def _compute_spread_speed(self):
        """计算 8 个方向的火势扩散速度"""
        angle_offset = [
            math.pi * 0.00,  # d=0 (正右 →, 0°)
            math.pi * 0.25,  # d=1 (右上 ↗, 45°)
            math.pi * 0.50,  # d=2 (正上 ↑, 90°)
            math.pi * 0.75,  # d=3 (左上 ↖, 135°)
            math.pi * 1.00,  # d=4 (正左 ←, 180°)
            math.pi * 1.25,  # d=5 (左下 ↙, 225°)
            math.pi * 1.50,  # d=6 (正下 ↓, 270°)
            math.pi * 1.75  # d=7 (右下 ↘, 315°)
        ]
        for x in range(1,self.grid_size[0]-1):
            for y in range(1,self.grid_size[1]-1):
                for d, (dx, dy) in enumerate(directions_8):  # 遍历8个方向
                    # **先检查是否在边界内**
                    if 0 < x + dx < self.grid_size[0]-1 and 0 < y + dy < self.grid_size[1]-1:
                        height_diff = abs(self.grid['height'][x, y] - self.grid['height'][x + dx, y + dy])
                    else:
                        height_diff = np.nan  # **超出边界，设为 NaN**
                        self.grid['spread_speed'][x, y][d] = np.nan
                        continue  # **跳过计算，防止 NaN 影响后续计算**

                    # **确保 height_diff 不为 0**
                    if height_diff == 0:
                        height_diff = 0.01  # 兜底，避免计算异常

                    # **计算风向影响**
                    wind_correction = math.cos(angle_offset[d] - self.wind_direction)
                    # wind_correction = 1.0
                    # **计算坡度修正**
                    divisor = (1.414 * 100) if d in {1, 3, 5, 7} else 100  # 斜向和正向区分
                    if self.grid['height'][x, y] > self.grid['height'][x + dx, y + dy]:  # 下坡
                        slope_factor = math.e ** (3.533 * -1 * height_diff / divisor)
                    else:  # 上坡
                        slope_factor = math.e ** (3.533 * height_diff / divisor)

                    # **计算最终传播速度**
                    self.grid['spread_speed'][x, y][d] = self.R0 * self.ks * math.e ** (
                            0.1783 * self.wind_speed * wind_correction) * slope_factor*3

                    # print(f"{x}，{y}: 火焰传播速度 {self.grid['spread_speed'][x, y]}")

    def update_burning_state(self, time_step=1.0):
        """更新燃烧状态，只有完全燃烧的格子才会传播火焰"""
        a = 100.0  # 格子边长 100m
        new_burning_state = self.grid['burning_state'].copy()  # 复制旧的燃烧状态

        for x in range(1,self.grid_size[0]-1):
            for y in range(1,self.grid_size[1]-1):
                state = self.grid['burning_state'][x, y]

                # ✅ 特殊状态直接跳过（已烧尽或被扑灭）
                if state == 2 or state == 3:
                    continue

                # ✅ 已完全燃烧，计时烧尽逻辑
                if state == 1.0:
                    self.burn_time_counter[x, y] += 1
                    if self.burn_time_counter[x, y] >= 50:
                        self.grid['burning_state'][x, y] = 2  # ✅ 烧尽
                    continue  # 不传播火焰，也不更新 new_burning_state
                else:
                    self.burn_time_counter[x, y] = 0  # ✅ 状态非完全燃烧，重置计时器

                # ✅ 火势传播更新（只对 0~1 状态执行）
                sum_R = 0
                for d in [0, 2, 4, 6]:  # 正方向
                    dx, dy = directions_8[d]
                    nx, ny = x + dx, y + dy
                    if 0 < nx < self.grid_size[0]-1 and 0 < ny < self.grid_size[1]-1:
                        if self.grid['burning_state'][nx, ny] == 1.0:
                            opposite_d = (d + 4) % 8
                            sum_R += self.grid['spread_speed'][nx, ny][opposite_d]
                term1 = (sum_R * time_step) / a

                sum_R2 = 0
                for d in [1, 3, 5, 7]:  # 对角方向
                    dx, dy = directions_8[d]
                    nx, ny = x + dx, y + dy
                    if 0 < nx < self.grid_size[0]-1 and 0 < ny < self.grid_size[1]-1:
                        if self.grid['burning_state'][nx, ny] == 1.0:
                            opposite_d = (d + 4) % 8
                            sum_R2 += self.grid['spread_speed'][nx, ny][opposite_d] ** 2
                term2 = (sum_R2 * time_step) / (2 * a ** 2)

                new_burning_state[x, y] = min(1.0, new_burning_state[x, y] + term1 + term2)

        mask_update = ~((self.grid['burning_state'] == 2) | (self.grid['burning_state'] == 3))
        self.grid['burning_state'][mask_update] = new_burning_state[mask_update]

    def compute_boundary_explore_rate(self, map_info):
        """
        真·边界：env.grid['burning_state'] 中，火格且八邻域至少一个为0（健康）
        已知边界：在这些边界火格上，map_info 已识别为火/烧尽(>0 或≈2)，
                且存在至少一个邻居在 map_info 中被识别为健康(=0)
        返回 (rate, true_cnt, known_cnt)
        """
        M = self.grid["burning_state"]
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
        known_fire = ((map_info > 0)&(map_info<=1)) | np.isclose(map_info, 2.0)

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

        # self.know_boundary_pre = self.known_boundary
        self.known_boundary = known_cnt
        self.true_boundary = true_cnt

    def _collect_fire_cells(self):
        """
        返回当前作为“灭火目标”的像素坐标列表。
        优先用 uav.B 的集合；若为空，则退化为 burning_state==1 的所有格子。
        """
        fire_cells = []
        for uav in getattr(self, "uavs", []):
            if hasattr(uav, "B") and isinstance(uav.B, (list, tuple)) and len(uav.B) > 0:
                fire_cells.extend(list(uav.B))
        if not fire_cells:
            M = self.grid["burning_state"]
            xs, ys = np.where(np.isclose(M, 1.0))
            fire_cells = list(zip(xs.tolist(), ys.tolist()))
        return fire_cells
    def _nearest_fire_dist(self, x: int, y: int, fire_cells):
        """
        给定 (x,y) 与 fire_cells(list[(i,j)])，返回曼哈顿最近距离；若无目标返回 +inf。
        """
        if not fire_cells:
            return float("inf")
        arr = np.abs(np.array(fire_cells) - np.array([[x, y]])).sum(axis=1)
        return float(arr.min())
    
    def _team_rewards_difference(self, options, kb_delta_total, fire_delta_total):
        """
        options: list[int]  本步每个 UAV 的高层任务(0=探索,1=灭火)
        返回: list[float]    每个 UAV 的逐步奖励（按效率分摊的差分式）
        仅用“最近火头距离 + 弹药”作为灭火效率；探索侧均分全局探索增量（不用未知边界距离）
        """
        import numpy as np
        eps = 1e-8

        # ---- 覆盖率与权重（保持你原先的风格，可微调为更平滑）----
        true_b  = max(1, int(getattr(self, "true_boundary", 0)))
        known_b = int(getattr(self, "known_boundary", 0))
        coverage = float(known_b) / float(true_b)
        TH_ATTACK, BAND_LOW = 0.95, 0.90

        # 探索权重：coverage->0.95 线性衰减到 0
        if coverage >= TH_ATTACK:
            w_explore = 0.0
        else:
            w_explore = np.clip((TH_ATTACK - coverage) / max(1e-6, (TH_ATTACK - BAND_LOW)), 0.0, 1.0)

        # 灭火权重：BAND_LOW 以下不鼓励；以上放大
        w_attack = 0.0 if coverage <= BAND_LOW else 2.0

        # ---- 集合与效率权重 ----
        fire_cells = self._collect_fire_cells()
        # 灭火效率：exp(-alpha * dist2fire) * g(ammo)
        alpha_f = 0.15
        fire_weights = []
        explore_flags = []
        for uav, opt in zip(self.uavs, options):
            if opt == 1:
                d_fire   = self._nearest_fire_dist(uav.x, uav.y, fire_cells)
                ammo_fr  = float(uav.d) / 10.0  # 假定 0~10 弹药
                eff_fire = np.exp(-alpha_f * d_fire) * (0.5 + 0.5 * ammo_fr)
                fire_weights.append(eff_fire)
                explore_flags.append(0)
                # print(f"无人机的灭火效率为{fire_weights},距离火点{d_fire}格")
            else:
                fire_weights.append(0.0)
                explore_flags.append(1)


        sum_wf = float(np.sum(fire_weights))
        n_expl = int(np.sum(explore_flags))

        # ---- 全局增量池 ----
        pool_explore = 0.1 * float(max(0, kb_delta_total)) * float(w_explore)
        pool_fire    = 0.2 * float(max(0, fire_delta_total)) * float(w_attack)

        # ---- 火头长度（用于无火惩罚）----
        head_len = 0
        if len(self.uavs) > 0 and hasattr(self.uavs[0], "B") and isinstance(self.uavs[0].B, (list, tuple)):
            head_len = len(self.uavs[0].B)

        # ---- 分配 ----
        rewards = []
        for uav, opt, w_f, expl_flag in zip(self.uavs, options, fire_weights, explore_flags):
            r = 0.0
            if opt == 0:
                # 探索侧：均分全局探索收益（你不想加 frontier 距离，就均分）
                if n_expl > 0 and pool_explore > 0:
                    r += pool_explore / n_expl
                r += 0.01  # 存在奖，防止纯0

                # 若 coverage 已高、有火且没人灭火 → 轻微惩罚
                if (coverage >= 0.90) and (head_len > 0) and (sum_wf <= eps):
                    r -= 0.2
            else:
                if head_len == 0:
                    r -= 0.2  # 无火还灭火
                else:
                    if sum_wf > eps and pool_fire > 0:
                        r += pool_fire * (w_f / sum_wf)
                    r += 0.01  # 存在奖
            rewards.append(float(r))
        return rewards
    
    def _team_rewards_difference_v2(
        self,
        options,
        kb_delta_total,           # 全局“新探索边界”增量
        head_delta_total,         # 全局“火头”减少量（>0 表示有效灭火）
        flank_delta_total,        # 全局“侧翼”减少量
        tail_delta_total,         # 全局“尾部”减少量
        cfg=None                  # 可选：系数配置字典
    ):
        """
        高层四任务差分奖励（探索/火头/侧翼/尾部）
        规则：
        - coverage < 0.90：只给探索，其他灭火权重=0
        - coverage >= 0.90：探索权重=0，火头/侧翼/尾部开启，
            每类奖励 = 任务权重 * 基础奖励值(集合长度×系数) * 个体效率权重
        返回：list[float]，每个UAV的即时奖励
        """
        import numpy as np
        eps = 1e-8

        # ---- 默认系数（可通过 cfg 覆盖）----
        C = {
            # 覆盖率阈值与放大
            "TH_ON": 0.90,      # >= 开始鼓励灭火
            "TH_HOLD": 0.80,
            "BAND_LOW": 0.90,   # 保留，兼容你之前逻辑（用不上也不冲突）

            # 探索与灭火的池子系数（整体缩放）
            "K_EXPLORE": 0.10,
            "K_HEAD":    0.20,
            "K_FLANK":   0.15,
            "K_TAIL":    0.10,

            # “基础奖励值”中各集合的长度系数（你提出的：头*2，侧翼*1，尾部*0.5）
            "BASE_HEAD_PERLEN":  2.0,
            "BASE_FLANK_PERLEN": 1.0,
            "BASE_TAIL_PERLEN":  0.5,

            # 个体效率权重：exp(-α·dist) * (0.5 + 0.5*ammo_frac)
            "ALPHA_DIST": 0.15,
            "AMMO_MAX":   10.0,

            # 覆盖率<阈值时的探索衰减（与旧逻辑一致线性衰减）
            "TH_ATTACK":  0.95,  # 旧名，兼容
            "BAND_LOW2":  0.90,  # 旧名，兼容

            # 存在奖（防止全0）
            "PRESENCE":   0,
            "INVALID_PENALTY": 0.4,
            "SWITCH_PENALTY": 0.2,
            "K_PROG": 0.08,
            "HOLD_BONUS_PER_STEP": 0.01,
        }
        if cfg is not None:
            C.update(cfg)

        # ---- 覆盖率与开关 ----
        true_b  = max(1, int(getattr(self, "true_boundary", 0)))
        known_b = int(getattr(self, "known_boundary", 0))
        coverage = float(known_b) / float(true_b)
        # ① 记录“之前是否已经到过阈值”
        was_on_prev = bool(getattr(self, "_has_reached_th_on", False))
        if not hasattr(self, "_has_reached_th_on"):
            self._has_reached_th_on = False
        if coverage >= C["TH_ON"]:
            self._has_reached_th_on = True
        # ② 本步是否“刚刚跨过阈值”（用于免除开局探索→灭火的切换惩罚）
        just_crossed_on = (not was_on_prev) and (coverage >= C["TH_ON"])
        # 探索期（<TH_ON）线性权重；达到阈值后探索=0，三类灭火开启
        if not self._has_reached_th_on:
            # 尚未达到过 0.90：只鼓励探索，不奖励三类灭火
            w_explore = np.clip(
                (C.get("TH_ATTACK", 0.95) - coverage) / max(1e-6, (C.get("TH_ATTACK", 0.95) - C.get("BAND_LOW2", 0.90))),
                0.0, 1.0
            )
            w_head = w_flank = w_tail = 0.0
        else:
            # 达到过 0.90：若仍>=0.80，继续奖励灭火；跌破0.80才关闭灭火奖励、转回鼓励探索
            if coverage >= C["TH_HOLD"]:
                w_explore = 0.5
                w_head = w_flank = w_tail = 1.0
            else:
                # 覆盖率掉到 0.80 以下：不奖励灭火，回到鼓励探索（线性衰减可按需调整）
                w_explore = np.clip(
                    (C.get("TH_ATTACK", 0.95) - coverage) / max(1e-6, (C.get("TH_ATTACK", 0.95) - C.get("BAND_LOW2", 0.90))),
                    0.0, 1.0
                )
                w_head = w_flank = w_tail = 0.0

        # ---- 各类集合长度（用第0架UAV维护的边界集合即可；如你有全局集合也可替换）----
        head_len  = len(self.uavs[0].B)        if (len(self.uavs)>0 and hasattr(self.uavs[0],"B"))        else 0
        flank_len = len(self.uavs[0].B_flank)  if (len(self.uavs)>0 and hasattr(self.uavs[0],"B_flank"))  else 0
        tail_len  = len(self.uavs[0].B_tail)   if (len(self.uavs)>0 and hasattr(self.uavs[0],"B_tail"))   else 0

        def _linear_scale(length):
            return float(np.clip(1.0 + 0.05 * float(length), 1.0, 2.0))
        
        # ---- 基础奖励值（与你设想一致）----
        base_head  = C["BASE_HEAD_PERLEN"]  * _linear_scale(head_len)
        base_flank = C["BASE_FLANK_PERLEN"] * _linear_scale(flank_len)
        base_tail  = C["BASE_TAIL_PERLEN"]  * _linear_scale(tail_len)

        # ---- 全局池（差分增量 × 池子系数 × 任务开关 × 基础奖励值）----
        pool_explore = C["K_EXPLORE"] * float(max(0, kb_delta_total)) * w_explore
        pool_head    = C["K_HEAD"]    * float(max(0, head_delta_total))  * w_head  * base_head
        pool_flank   = C["K_FLANK"]   * float(max(0, flank_delta_total)) * w_flank * base_flank
        pool_tail    = C["K_TAIL"]    * float(max(0, tail_delta_total))  * w_tail  * base_tail

        # ---- 个体效率权重（只对非探索选项计算）----
        # 若未提供 dist_to_target_list，则按任务就近到集合的最小距离估计
        def _nearest_dist_to(points, ux, uy):
            if not points:
                return 1e6
            dmin = 1e6
            for (x,y) in points:
                d = np.hypot(ux - x, uy - y)
                if d < dmin: dmin = d
            return dmin

        alpha = C["ALPHA_DIST"]
        ammo_max = max(1.0, float(C["AMMO_MAX"]))

        eff_head  = []
        eff_flank = []
        eff_tail  = []

        # 收集三类效率（与 options 对齐）
        for i, (uav, opt) in enumerate(zip(self.uavs, options)):
            
            if opt == 0:
                eff_head.append(0.0); eff_flank.append(0.0); eff_tail.append(0.0)
                continue

            elif opt == 1:
                d_fire = _nearest_dist_to(getattr(self.uavs[0], "B", []), uav.x, uav.y)
            elif opt == 2:
                d_fire = _nearest_dist_to(getattr(self.uavs[0], "B_flank", []), uav.x, uav.y)
            else: # opt == 3
                d_fire = _nearest_dist_to(getattr(self.uavs[0], "B_tail", []), uav.x, uav.y)

            ammo_fr = float(getattr(uav, "d", 0.0)) / ammo_max  # 假设 d 是弹药数
            eff = np.exp(-alpha * d_fire) * (0.5 + 0.5 * np.clip(ammo_fr, 0.0, 1.0))

            if opt == 1:
                eff_head.append(eff);  eff_flank.append(0.0); eff_tail.append(0.0)
            elif opt == 2:
                eff_head.append(0.0);  eff_flank.append(eff); eff_tail.append(0.0)
            elif opt == 3:
                eff_head.append(0.0);  eff_flank.append(0.0); eff_tail.append(eff)
            else:
                eff_head.append(0.0);  eff_flank.append(0.0); eff_tail.append(0.0)

        sum_eh = float(np.sum(eff_head))
        sum_ef = float(np.sum(eff_flank))
        sum_et = float(np.sum(eff_tail))

        # 统计探索人数
        n_expl = int(np.sum([1 if o == 0 else 0 for o in options]))

        # ---- 最终分配 ----
        rewards = []
        for i, (uav, opt) in enumerate(zip(self.uavs, options)):
            r = 0.0
            
            illegal = ((opt == 1 and head_len  == 0) or
               (opt == 2 and flank_len == 0) or
               (opt == 3 and tail_len  == 0))
            if illegal:
                r -= C.get("INVALID_PENALTY", 0.2)   # 明确的“选错任务”惩罚

            # --- 任务切换惩罚 ---
            prev_opt = getattr(uav, "prev_option", None)
            if (prev_opt is not None) and (opt != prev_opt):
                # 如果不是因为集合空了才切换，则惩罚
                was_empty = ((prev_opt == 1 and self.head_len_pre == 0) or
                             (prev_opt == 2 and self.flank_len_pre == 0) or
                             (prev_opt == 3 and self.tail_len_pre == 0))
                
                # 开局探索→首次跨过阈值开始灭火：免罚
                exploring_to_fire_first_time = (prev_opt == 0) and (opt in (1, 2, 3)) and just_crossed_on
                # 覆盖率跌破 TH_HOLD 被迫回到探索：免罚（可选，但推荐保留）
                forced_hysteresis = (prev_opt in (1, 2, 3)) and (opt == 0) and (coverage < C["TH_HOLD"])

                if not (was_empty or exploring_to_fire_first_time or forced_hysteresis):
                    r -= C.get("SWITCH_PENALTY", 0.3)
                    
            uav.prev_option = opt


            # --- 任务承诺奖励 ---
            hold = getattr(uav, "option_hold", 0)
            if (prev_opt is not None) and (opt == prev_opt):
                hold = min(hold + 1, 5)
            else:
                hold = 0
            r += C.get("HOLD_BONUS_PER_STEP", 0.01) * hold
            uav.option_hold = hold


            if opt == 0:
                # 探索：均分探索池 + 存在奖
                if n_expl > 0 and pool_explore > 0:
                    r += pool_explore / n_expl
                r += C["PRESENCE"]

                # 若 coverage 已高且三类池都几乎没人拿，则轻微惩罚（可选）
                if (coverage >= C["TH_ON"]) and (sum_eh + sum_ef + sum_et <= eps):
                    r -= 0.2

            elif opt == 1:
                # 火头：按效率分配 + 存在奖
                if sum_eh > eps and pool_head > 0:
                    r += pool_head * (eff_head[i] / sum_eh)
                r += C["PRESENCE"]

            elif opt == 2:
                if sum_ef > eps and pool_flank > 0:
                    r += pool_flank * (eff_flank[i] / sum_ef)
                r += C["PRESENCE"]

            elif opt == 3:
                if sum_et > eps and pool_tail > 0:
                    r += pool_tail * (eff_tail[i] / sum_et)
                r += C["PRESENCE"]

            r -=0.1
            rewards.append(float(r))

        return rewards

    def compute_agent_reward(self, option, kb_delta, fire_diff,frac=None):
        # ---- 1) 覆盖率 ----
        true_b = max(1, int(getattr(self, "true_boundary", 0)))
        known_b = int(getattr(self, "known_boundary", 0))
        coverage = float(known_b) / float(true_b)

        # ---- 2) 火头长度 ----
        head_len = 0
        if hasattr(self, "uavs") and len(self.uavs) > 0 and hasattr(self.uavs[0], "B") and isinstance(self.uavs[0].B, (list, tuple)):
            head_len = len(self.uavs[0].B)

        # ---- 3) 增量 ----
        diff_explore = kb_delta
        fire_diff    = fire_diff

        # ---- 4) 权重与阈值（沿用你的）----
        TH_ATTACK = 0.95
        BAND_LOW  = 0.90

        # 探索权重
        if coverage >= TH_ATTACK:
            w_explore = 0
        else:
            w_explore = np.clip((TH_ATTACK - coverage) / max(1e-6, (TH_ATTACK - BAND_LOW)), 0.0, 1.0)

        # 灭火权重
        if coverage <= BAND_LOW:
            w_attack = 0.0
        else:
            w_attack = 2.0
       

        # ---- 5) head_len -> K（需要灭火的人数）----
        if head_len == 0:
            K_fire = 0
        elif head_len <= 5:
            K_fire = 1
        elif head_len <= 10:
            K_fire = 2
        else:
            K_fire = 3

        # 当前灭火人数（本 UAV 决策之前的“已选”）
        fire_count_now = frac[1]
        reward = 0.0

        if option == 0:  # ---- 探索 ----
            reward += 0.1 * diff_explore * w_explore +0.02

            # 仅在：覆盖率已达阈值 + 仍有火头 + 当前灭火人数 < K 时，才惩罚“还去探索”
            if (coverage >= 0.9) and (head_len > 0):
                if fire_count_now < K_fire:
                    # 灭火缺员而仍探索 => 小惩罚（维持你原逻辑）
                    reward -= 0.2
                    # print(f"目前选的人数是{fire_count_now},需要的无人机数量为{K_fire}")
                else:
                    # 灭火已达标 => 给探索一个“待命/巡逻”小正奖，避免全员扑向灭火
                    reward += 0.05

            # 若 head_len==0（火头已灭），探索无此惩罚

        elif option == 1:  # ===== 灭火 =====
            base = 0.2 * fire_diff * w_attack

            if head_len == 0:
                # 没火还灭火
                reward += base - 0.2
                return float(reward)

            fire_count_after = fire_count_now + 1
            over = max(0, fire_count_after - K_fire)

            if over <= 0:
                # 未超编：拿到完整基础灭火奖励
                reward += base
                if fire_diff <= 0:
                    reward += 0.02
            else:
                # 超编：强拥挤折扣 + 显式线性惩罚
                discount = max(0.3, 1.0 - 0.2 * over)
                crowded_reward = base * discount - 0.15 * over
                # 设一个小的下限，避免过大负数
                reward += crowded_reward

            

        return float(reward)



    def write_local_fire_to_uav_map(self, uav):
        # 观测半径：未见火 3x3，见火 5x5
        r = 2 if uav.memory_flag else 1
        x, y = uav.x, uav.y
        H, W = self.grid["burning_state"].shape

        xs = max(0, x - r)
        xe = min(H, x + r + 1)
        ys = max(0, y - r)
        ye = min(W, y + r + 1)

        patch = self.grid["burning_state"][xs:xe, ys:ye]
        # 把“看见的”区域抄到自己的地图；未看的仍保持 -1
        uav.map_info[xs:xe, ys:ye] = patch

    def fuse_team_maps_max(self):
        """把所有 UAV 的 map_info 融合为一个，并回写到每个 UAV。"""
        # 假设所有 uav.map_info 形状一致
        maps = [uav.map_info for uav in self.uavs]
        fused = np.maximum.reduce(maps)          # 等价于逐格取 max
        for uav in self.uavs:
            uav.map_info[...] = fused            # 回写（共享同一张态势图）

    def step(self, actions):
        """
        多无人机step函数
        输入:
            actions: list[int]，每个无人机的动作索引
        输出:
            observations: list[obs]
            rewards: list[float]
            terminated: bool
            truncated: bool
            infos: dict
        """
        rewards = [0.0 for _ in self.uavs]
        terminated = False
        truncated = False
        infos = {}

        pre_state = self.grid["burning_state"].copy()  # 本步动作/蔓延前的状态

        pre_head_set  = set(getattr(self.uavs[0], "B", []) or [])
        pre_flank_set = set(getattr(self.uavs[0], "B_flank", []) or [])
        pre_tail_set  = set(getattr(self.uavs[0], "B_tail", []) or [])


        # 1️⃣ 先更新位置+火点
        for idx, (uav, action) in enumerate(zip(self.uavs, actions)):
            if action < 8:
                dx, dy = [
                    (1,0), (1,1), (0,1), (-1,1), (-1,0),
                    (-1,-1), (0,-1), (1,-1)
                ][action]
                new_x = uav.x + dx
                new_y = uav.y + dy
                if 0 < new_x < self.grid_size[0]-1 and 0 <= new_y < self.grid_size[1]-1:
                    uav.x = new_x
                    uav.y = new_y

                    if uav.x == uav.current_target[0] and uav.y == uav.current_target[1]:
                        self.grid['burning_state'][uav.x, uav.y] = 3.0
            else:
                #原地停留
                if uav.x == uav.current_target[0] and uav.y == uav.current_target[1]:
                        self.grid['burning_state'][uav.x, uav.y] = 3.0

        # 2️⃣ 更新火焰
        if self.fire_update_counter % 10 == 0:
            self.update_burning_state(time_step=1)

        self.fire_update_counter += 1

        #更新观测
        for uav in self.uavs:
            self.write_local_fire_to_uav_map(uav)
        #更新全队地图
        self.fuse_team_maps_max()
        # 3️⃣ 获取新观测
        observations = []
        all_positions = {u.uav_id: (u.x, u.y) for u in self.uavs}
        for uav in self.uavs:
            obs = self.get_observation(
                uav=uav,
                burning_state=self.grid["burning_state"],
                q_fire=(25,25),
                all_uav_positions=all_positions
            )
            observations.append(obs)

        #计算全局观测增量
        kb_prev = int(getattr(self, "know_boundary_pre", 0))
        self.compute_boundary_explore_rate(self.uavs[0].map_info)  # 只算一次，更新 self.known_boundary
        kb_delta = max(0, int(self.known_boundary) - kb_prev)

        #计算全局灭火增量
        extinguish_now = int(np.count_nonzero(self.grid["burning_state"] == 3))
        fire_diff = extinguish_now - int(getattr(self, "extinguish_pre", 0))

        post = self.grid["burning_state"]
        new_ext_mask = ((pre_state > 0) & (pre_state <= 1)) & (post == 3)

        # 逐类统计（只统计“本步新灭”的那些格子）
        head_delta  = 0
        flank_delta = 0
        tail_delta  = 0

        # 遍历变为3.0的格子；也可以用 np.where 提速（地图不大时直接遍历够用）
        xs, ys = np.where(new_ext_mask)
        for x, y in zip(xs, ys):
            p = (int(x), int(y))
            if p in pre_head_set:
                head_delta += 1
            elif p in pre_flank_set:
                flank_delta += 1
            elif p in pre_tail_set:
                tail_delta += 1

         # === 在这里记录本步前的长度，用于奖励函数里的切换惩罚 ===
        self.head_len_pre = len(pre_head_set)
        self.flank_len_pre = len(pre_flank_set)
        self.tail_len_pre = len(pre_tail_set)

        # === 记录每个 UAV 到三个类别的最近距离，用于进展奖励 ===
        def nearest_dist(uav, points):
            if not points: return None
            ux, uy = int(uav.x), int(uav.y)
            return min(abs(ux-x)+abs(uy-y) for (x,y) in points)

        for uav in self.uavs:
            uav.dist_prev = {
                "head":  nearest_dist(uav, list(pre_head_set)),
                "flank": nearest_dist(uav, list(pre_flank_set)),
                "tail":  nearest_dist(uav, list(pre_tail_set)),
            }
        

        options_now = [int(uav.option) for uav in self.uavs]
        # rewards = self._team_rewards_difference(options_now, kb_delta, fire_diff)

        rewards = self._team_rewards_difference_v2(options_now, kb_delta, head_delta, flank_delta, tail_delta, cfg=None)
        self.know_boundary_pre = int(self.known_boundary)
        self.extinguish_pre = extinguish_now
        
        # 5️⃣ 终止条件
        explore_rate = float(self.known_boundary/self.true_boundary)

        if (len(self.uavs[0].B) == 0 and len(self.uavs[0].B_flank) == 0 and len(self.uavs[0].B_tail) == 0 and explore_rate >=0.99) or self.fire_update_counter >=300:
            terminated = True

        return observations, rewards, terminated, truncated, infos





    def render(self, mode="human", pause_time=0.1):
        #转置操作：numpy矩阵默认第一位是行,第二位是列，在渲染的时候我首先交换了行和列，并且进行了上下颠倒，使得渲染出来是数学中的第一象限
        if mode != "human":
            return

        # 初始化 RGB 色图为白色
        color_map = np.ones((*self.grid_size, 3))  # 默认白色背景

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                burning_state = self.grid['burning_state'][x, y]

                if burning_state == 0:
                    color_map[y, x] = [1, 1, 1]  # 探索频率 → 蓝白渐变
                elif 0 < burning_state < 1:
                    color_map[y, x] = [1.0, 0.5, 0.0]  # 燃烧中：橙色
                elif burning_state == 1.0:
                    color_map[y, x] = [1.0, 0.0, 0.0]  # 完全燃烧：红色
                elif np.isclose(burning_state, 2.0):
                    color_map[y, x] = [0.0, 0.0, 0.0]  # 烧尽：黑色
                elif np.isclose(burning_state, 3.0):
                    color_map[y, x] = [0.0, 1.0, 0.0]  # 被扑灭：绿色

        # 绘制 UAV：紫色
        for uav in self.uavs:
            if 0 <= uav.x < self.grid_size[0] and 0 <= uav.y < self.grid_size[1]:
                color_map[uav.y, uav.x] = [0.6, 0.0, 0.8]

        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.img = self.ax.imshow(color_map, origin='lower')
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        else:
            self.img.set_data(color_map)

        [child.remove() for child in self.ax.patches[:]]  # 删除之前所有方框

        # ✅ 绘制目标点为蓝色方框（坐标也需交换）
        for uav in self.uavs:
            if uav.current_target:
                tx, ty = uav.current_target
                if 0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]:
                    rect = plt.Rectangle((tx - 0.5, ty - 0.5), 1, 1, linewidth=1.5,
                                        edgecolor='blue', facecolor='none')
                    self.ax.add_patch(rect)

            if hasattr(uav, "B") and uav.B:
                for bx, by in uav.B:
                    if 0 <= bx < self.grid_size[0] and 0 <= by < self.grid_size[1]:
                        rect = plt.Rectangle(
                            (bx - 0.5, by - 0.5), 1, 1,
                            linewidth=1.0, edgecolor='yellow', facecolor='none'
                        )
                        self.ax.add_patch(rect)
            
            if hasattr(uav, "B_flank") and uav.B_flank:
                for bx, by in uav.B_flank:
                    if 0 <= bx < self.grid_size[0] and 0 <= by < self.grid_size[1]:
                        rect = plt.Rectangle(
                            (bx - 0.5, by - 0.5), 1, 1,
                            linewidth=1.0, edgecolor='black', facecolor='none'
                        )
                        self.ax.add_patch(rect)
            if hasattr(uav, "B_tail") and uav.B_tail:
                for bx, by in uav.B_tail:
                    if 0 <= bx < self.grid_size[0] and 0 <= by < self.grid_size[1]:
                        rect = plt.Rectangle(
                            (bx - 0.5, by - 0.5), 1, 1,
                            linewidth=1.0, edgecolor='green', facecolor='none'
                        )
                        self.ax.add_patch(rect)

        self.ax.set_title(f"🔥 Step: {self.fire_update_counter}", fontsize=14)
        self.fig.canvas.draw_idle()
        plt.pause(pause_time)

    def save_fire_map_image(self,map_info, save_path="fire_map.png"):
        """
        保存火势地图图片：
        - 未探索 (-1): 黑色
        - 未着火 (0): 白色
        - 着火 (>0): 红色
        """
        H, W = map_info.shape
        img = np.zeros((H, W, 3), dtype=np.uint8)

        for x in range(H):
            for y in range(W):
                if map_info[x, y] == -1:
                    img[x, y] = [0, 0, 0]       # 黑色
                elif map_info[x, y] == 0:
                    img[x, y] = [255, 255, 255] # 白色
                elif map_info[x, y] > 0:
                    img[x, y] = [255, 0, 0]     # 红色

        plt.imsave(save_path, img)



    
    def close(self):
        """关闭环境"""
        pass

