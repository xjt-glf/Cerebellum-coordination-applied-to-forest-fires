import math
import random
import gymnasium as gym  # ✅ 使用 gymnasium
from gymnasium import spaces  # ✅ 确保 spaces 也从 gymnasium 导入
import numpy as np
# matplotlib 仅用于 render()/save_fire_map_image() 的可视化，不参与物理演化与
# 训练/评估。容器环境（H200 训练）无 matplotlib 时允许不装：render 会显式报错。
try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:  # pragma: no cover
    plt = None
from collections import deque
# pandas 仅用于 reset() 时读 terrain_height.csv（地形高程）。容器训练环境无
# pandas 时用 csv 模块等价解析（同一文件、同一列结构），物理行为不变。
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None
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
        self.unknown = []

        self.just_dropped = False

        self.edge_dir = None
        
        # 标记是否正在执行通信交换任务（用于无人机1、2）
        self.is_exchanging_info = False
        
        # 距离上一次与人员通信的时间（用于高层调度）
        self.time_since_last_comm = 0

        # ===== 低层任务条件策略训练所需的逐步事件标记（lowlevel/ 包使用）=====
        self.e_max = float(e) if e else 300.0   # 电量上限，用于归一化
        self.blocked = False                    # 本步移动被其它无人机阻挡
        self.oob = False                        # 本步动作会越界，被环境丢弃
        self.recharged = False                  # 本步到达补给点并完成补给
        self.depleted = False                   # 电量耗尽
        self.new_known_cells = 0                # 本步新观测到的格子数
        self.new_boundary_cells = 0             # 本步新观测到的火线边界格子数
        self.exclusive_covered = 0              # 本步仅由本机覆盖的救援人员数
        self.shared_covered = 0                 # 本步由本机覆盖的救援人员数（含重叠）

class Rescuer:
    """地面救援人员类"""
    def __init__(self, rescuer_id, x, y):
        self.rescuer_id = rescuer_id  # 救援人员 ID
        self.x = x  # 当前位置（浮点数，支持连续移动）
        self.y = y
        self.status = "idle"  # 状态："moving_to_fire_head", "moving_to_side", "idle"
        self.target_pos = None  # 目标位置 (x, y)
        self.AoI = 0  # 信息时效（Age of Information）：自上次被无人机覆盖后的时间步数
        self.is_covered = False  # 当前是否在通信覆盖范围内
        self.move_counter = 0  # 移动计数器：每两步移动1格
        # 路径规划缓存（用于减少重复BFS计算）
        self.cached_path = None  # 缓存的路径 [(x1,y1), (x2,y2), ...]
        self.cached_path_target = None  # 缓存路径对应的目标点
        self.target_steps = 0  # 目标保持的步数（用于5步自动变换规则）
        self.cluster_anchor = None  # 负责的火点质心（多火点分散分配用）
        self.perimeter_dir = 1      # 绕火场外圈的方向（+1 / -1）
        # 路径规划缓存（用于减少重复BFS计算）
        self.cached_path = None  # 缓存的路径 [(x1,y1), (x2,y2), ...]
        self.cached_path_target = None  # 缓存路径对应的目标点

class GridEnv(gym.Env):
    """
    自定义的网格环境，模拟无人机群搜索救火任务。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    # 旧的 50x50 单火点场景，仅用于复现历史实验：GridEnv(**GridEnv.LEGACY_50x50)
    LEGACY_50x50 = dict(grid_size=(50, 50), n_fire_seeds=1, spread_gain=2.0,
                        fire_seed_hw=(8, 8), sense_radius=3, uav_energy=300.0,
                        uav_comm_radius=8, base_comm_radius=10,
                        charging_stations=[(5, 5), (5, 45), (45, 45), (45, 5)],
                        rescuer_start=(1.0, 1.0))

    def __init__(self, render_mode: str = "human", grid_size=(100, 100),
                 n_fire_seeds=3, spread_gain=1.0, fire_seed_hw=(4, 4),
                 spread_steps=40, sense_radius=4, num_uavs=4, num_rescuers=8,
                 uav_energy=600.0, uav_comm_radius=12, base_comm_radius=15,
                 charging_stations=None, rescuer_start=None,
                 rescuer_speed=0.5, extinguish_radius=2, extinguish_rate=2,
                 randomize_wind=True, n_fire_seeds_range=(2, 4), charge_radius=1):
        super(GridEnv, self).__init__()

        # 任务区域：默认 100x100 网格，一格 100m x 100m
        self.grid_size = tuple(grid_size)
        H, W = self.grid_size
        # 补给点：四角内缩 10% + 四边中点，共 8 个。
        # 只放四角时地图中心到最近站点约 55 格、往返约 110 步，而单次续航 240 步、
        # 回合 300 步，稳态下常有 2/4 台在通勤，覆盖率被结构性地压住。
        # 加上四边中点后最坏情况的单程距离从 ~55 格降到 ~40 格。
        self.charging_stations = list(charging_stations) if charging_stations else [
            (int(0.1 * H), int(0.1 * W)), (int(0.1 * H), int(0.9 * W)),
            (int(0.9 * H), int(0.9 * W)), (int(0.9 * H), int(0.1 * W)),
            (int(0.1 * H), int(0.5 * W)), (int(0.5 * H), int(0.1 * W)),
            (int(0.5 * H), int(0.9 * W)), (int(0.9 * H), int(0.5 * W))]
        # 补给点的有效半径（切比雪夫）。0 = 必须精确落在那一格。
        # 默认 1，即把补给点看作 3x3 的停机坪（一格 100m，停机坪 300m 见方）。
        # 原来是"精确一格"：无人机在站点旁边一格耗尽后会被 env_wrapper 强制悬停，
        # 从此永久卡死在 e=0，既动不了也充不上电。
        self.charge_radius = int(charge_radius)
        # 初始火点：n_fire_seeds 个分散的小火点，位置每回合随机（见 _initialize_grid）
        self.n_fire_seeds = int(n_fire_seeds)
        self.fire_seed_hw = tuple(fire_seed_hw)
        # spread_gain 是扩散速度的总增益：历史实现在公式末尾硬编码 *2，
        # 现在提出来做参数，默认 1.0 即"砍掉一半"。
        self.spread_gain = float(spread_gain)
        self.sense_radius = int(sense_radius)
        self.base_comm_radius = float(base_comm_radius)
        self.base_pos = (0.0, W / 2.0)          # 基站位置（地图左侧中点）
        self.share_all_observations = True      # 全队共享观测（配合集中式规划）
        # 人员只知道"自己八邻域看到的 + 无人机上传的"火情，不再直接读真值火图。
        # 置为 False 可退回历史行为（人员凭真值找火点，等于无人机白干）。
        self.rescuer_knows_only_shared_map = True
        self.rebalance_every = 25        # 每多少步把人员按名额重新摊到各活火点
        self.compute_env_rewards = False        # 见 step() 末尾的说明
        self.team_map_info = None               # 全队融合后的火势认知图
        self.rescuer_start = tuple(rescuer_start) if rescuer_start else (1.0, W / 2.0)
        # 人员每步走 rescuer_speed 格（0.5 = 两步一格），位置是浮点
        self.rescuer_speed = float(rescuer_speed)
        # 靠近即自动灭火：切比雪夫半径内的燃烧格，每人每步最多扑灭 extinguish_rate 个
        self.extinguish_radius = int(extinguish_radius)
        self.extinguish_rate = int(extinguish_rate)
        # 每回合随机风向 / 火点数：风向固定的话策略会记住"火往右跑"，换个风向就失效
        self.randomize_wind = bool(randomize_wind)
        self.n_fire_seeds_range = (tuple(n_fire_seeds_range)
                                   if n_fire_seeds_range else None)

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

        self.fire_update_counter = 0
        self.spread_steps  = int(spread_steps)
        self.init_hw = self.fire_seed_hw
        self.num_uavs = int(num_uavs)
        self.num_rescuers = int(num_rescuers)  # 救援人员数量（可配置）
        self.uav_energy = float(uav_energy)
        self._wind_text = None   # 占位

        # 通信相关参数
        self.uav_comm_radius = float(uav_comm_radius)  # 无人机通信半径（覆盖救援人员）

        # ===== 能耗模型 =====
        # 机队是同构的（所有无人机 e=300, d=5），因此每步能耗必须相同。
        # 历史实现用 `uav.e -= idx+1`，即第 i 架每步耗 i+1 —— 这会让各机在不同
        # 时刻耗尽，从物理层面"制造"出分批充电的假象，而非由策略学出来。
        # 现改为统一常数；取 2.5 = mean(1,2,3,4)，保持 4 机编队的机队总能耗与
        # 历史实验一致，从而旧的高层实验数值仍可比较。
        self.energy_cost = 2.5
        # "legacy_index" 仅为复现历史结果保留，不具备物理依据，请勿用于新实验。
        self.energy_cost_mode = "uniform"
        
        # 全局通信特征（用于高层观测）
        self.coverage_rate = 0.0  # 无人机对救援人员的通信覆盖率
        self.comm_coverage_rate = 0.0  # 有通信的人员比例（无人机或基站）
        self.max_AoI = 0  # 历史最大信息时效（诊断用）
        self.cur_max_AoI = 0  # 当前最大信息时效（奖励/观测用）
        
        # 创建无人机群
        self._initialize_uavs()
        
        # 创建救援人员（在 reset 中初始化）
        self.rescuers = []
        
        # 救援人员共享的火势地图（类似UAV的map_info，但名称不同）
        self.rescuers_map_info = None  # 将在reset中初始化为与grid_size相同的数组
        
        # 救援人员共享的火场结构集合（B_HEAD, B_FLANK, B_TAIL）
        self.rescuers_B_HEAD = []   # 火头集合
        self.rescuers_B_FLANK = []  # 侧翼集合
        self.rescuers_B_TAIL = []   # 尾部集合
        
        # ========== 目标分配策略的可调参数 ==========
        # 评分权重
        self.target_w_struct = 1.0   # 结构项权重
        self.target_w_dist = 1.0    # 距离项权重
        self.target_w_repulse = 2.0 # 排斥项权重
        
        # 距离项参数
        self.target_d0 = 10.0       # 距离归一化参数（用于距离项计算）
        
        # 结构项分数（根据火场结构类型）
        self.target_score_head = 3.0   # 火头分数
        self.target_score_flank = 2.0  # 侧翼分数
        self.target_score_tail = 1.0   # 尾部分数
        self.target_score_other = 0.0  # 其他类型分数
        
        # 排斥项参数
        self.target_repulse_radius = 3.0  # 排斥半径R（如果两个目标点距离<=R，则施加排斥）
        self.target_repulse_penalty = -3.0  # 排斥罚分（固定负值）

        # 初始化网格属性
        self._initialize_grid()

        # 定义动作空间
        self.action_space = spaces.Discrete(9)

        # 其他初始化
        self.render_mode = render_mode

        # 通信距离（用于部分可通信的火势地图交换）
        self.communication_radius = 10

        #高层记录K步前的火灾点
        self.fire_cells_prev = 0
        #记录火点差值
        self.fire_diff = 0

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
        self.option_counts = np.zeros(5, dtype=np.int32)
        #是否到达覆盖率
        self._has_reached_th_on = False

        self.head_len_pre  = 0    # 上一时刻的三类长度（切换惩罚中的“是否集合已空”判断）
        self.flank_len_pre = 0
        self.tail_len_pre  = 0

        
    def reset(self):
        # self.wind_level = int(np.random.randint(4, 7))
        # self.wind_level = 6
        # self.wind_speed = float(self.wind_level * 2)
        # 每回合随机风向与火点数（必须在 _initialize_grid 之前定好：
        # 扩散速度是在 _initialize_grid 里按风向算的）
        if getattr(self, "randomize_wind", False):
            self.wind_direction = float(np.random.uniform(0.0, 2.0 * math.pi))
        if getattr(self, "n_fire_seeds_range", None):
            lo, hi = self.n_fire_seeds_range
            self.n_fire_seeds = int(np.random.randint(int(lo), int(hi) + 1))

        # 重算与风相关的参数（供扩散速度使用）
        self.R0 = 0.029 * self.temperature + 0.047 * self.wind_level - 0.009 * self.humidity + 0.304
        self.kw = math.e ** (0.1783 * self.wind_speed)

        self._initialize_grid()
        self.fire_update_counter = 0
        self._initialize_uavs()
        self._initialize_rescuers()  # 初始化救援人员
        self._wind_text = None
        
        # 重置通信特征
        self.coverage_rate = 0.0
        self.max_AoI = 0
        self.cur_max_AoI = 0
        
        # 标志：是否曾经为救援人员分配过目标（用于结束条件判断）
        self.rescuers_had_target = False
        self.team_map_info = None
        self.explored_rate = 0.0
        self.fire_discovery_rate = 0.0
        self.burned_area_rate = float((self.grid["burning_state"] > 0).mean())
        self.rescuer_known_fire_rate = 0.0
        
        for uav in self.uavs:
            # 用全局火图同尺寸
            H, W = self.grid["burning_state"].shape
            uav.map_info = np.full((H, W), -1.0, dtype=np.float32)
            # uav.map_info[25,25] = 1
            # 重置通信交换标记
            uav.is_exchanging_info = False
            # 重置通信时间
            uav.time_since_last_comm = 0
        
        # 初始化所有救援人员的 AoI 和覆盖状态
        H, W = self.grid_size
        for rescuer in self.rescuers:
            rescuer.AoI = 0
            rescuer.is_covered = False
            rescuer.status = "idle"
            rescuer.target_pos = None
            rescuer.move_counter = 0  # 重置移动计数器
            rescuer.target_steps = 0  # 重置目标保持步数计数器
            rescuer.cluster_anchor = None  # 负责的火点（多火点分散分配用）
            rescuer.perimeter_dir = 1 if rescuer.rescuer_id % 2 else -1  # 绕行方向
            # 清除路径规划缓存
            rescuer.cached_path = None
            rescuer.cached_path_target = None
        
        # 初始化救援人员共享的火势地图（初始值为-1表示未知）
        self.rescuers_map_info = np.full((H, W), -1.0, dtype=np.float32)
        
        # 重置救援人员共享的火场结构集合
        self.rescuers_B_HEAD = []
        self.rescuers_B_FLANK = []
        self.rescuers_B_TAIL = []
        
        # 收集所有无人机观测
        observations = []
        all_positions = {u.uav_id: (u.x, u.y) for u in self.uavs}
        # print(all_positions)
        for uav in self.uavs:
            obs = self.get_observation(
                uav=uav,
                burning_state=self.grid["burning_state"],
                q_fire=(15,25),
                all_uav_positions=all_positions
            )
            observations.append(obs)

        self.fire_cells_prev = int(np.count_nonzero(self.grid["burning_state"] > 0))
        return observations, {}


    def _initialize_grid(self):
        """初始化网格的四个参数"""
        self.grid['flammability'] = np.random.uniform(0.2, 1.0, self.grid_size)
        try:
            if pd is not None:
                height_df = pd.read_csv("terrain_height.csv", header=0)
                height_array = height_df.values.astype(np.float32)
            else:
                # 等价解析（无 pandas 容器）：同文件同列结构，数值一致
                import csv as _csv
                with open("terrain_height.csv", newline="") as _f:
                    _rows = list(_csv.reader(_f))
                height_array = np.asarray(
                    [[float(v) for v in row] for row in _rows[1:]], dtype=np.float32)
            if height_array.shape != self.grid_size:
                # 地形图是 50x50 的实测数据，任务区域放大后按最近邻重采样，
                # 保持"同一片地形"而不是换成随机高度（坡度会直接影响扩散速度）。
                src_h, src_w = height_array.shape
                xi = (np.arange(self.grid_size[0]) * src_h // self.grid_size[0]).clip(0, src_h - 1)
                yi = (np.arange(self.grid_size[1]) * src_w // self.grid_size[1]).clip(0, src_w - 1)
                height_array = height_array[np.ix_(xi, yi)]
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


        # 初始火点：n_fire_seeds 个互相分散的小火块，位置每回合随机。
        # 无人机的首要任务就是"把这些火点找出来"，所以位置必须是未知且变化的，
        # 否则策略会记住固定坐标，探索行为无从谈起。
        H, W = self.grid_size
        h, w = int(self.init_hw[0]), int(self.init_hw[1])
        margin = max(6, 2 * max(h, w))
        min_sep = max(h, w) * 4 + int(0.2 * min(H, W))   # 火点之间的最小间距
        seeds = []
        for _ in range(500):
            if len(seeds) >= self.n_fire_seeds:
                break
            cx = int(np.random.randint(margin, H - margin - w))
            cy = int(np.random.randint(margin, W - margin - h))
            if all(abs(cx - sx) + abs(cy - sy) >= min_sep for sx, sy in seeds):
                seeds.append((cx, cy))
        while len(seeds) < self.n_fire_seeds:     # 兜底：间距约束放宽
            seeds.append((int(np.random.randint(margin, H - margin - w)),
                          int(np.random.randint(margin, W - margin - h))))
        self.fire_seeds = seeds
        fire_position = [(x, y) for cx, cy in seeds
                         for x in range(cx, cx + w) for y in range(cy, cy + h)]
        # fire_position = [(i, j) for i in range(20, 30) for j in range(20, 30)]
        # fire_position = [(i, j) for i in range(20, 40) for j in range(20, 25)]

        # fire_position = [(15,25),(15,24),(14,24),(14,25)]
        for x, y in fire_position:
            self.grid['burning_state'][x, y] = 1.0
        # print(f"点燃格子{fire_position}")
        # spread_steps = np.random.randint(30, 60)   # 包含 15 和 30
        spread_steps = self.spread_steps
        # spread_steps = 40   # 包含 15 和 30
        for _ in range(spread_steps):
            self.update_burning_state(time_step=1)
            # self.render()
    def _initialize_uavs(self):
        self.uavs = []
        spawn_points = [
            (10, 10),     # 第1架
            (10, 0),    # 第2架
            (40, 40),     # 第3架
            (8,8),
            (40,40),
            (10,40)
        ]
        for uav_id in range(1, self.num_uavs+1):
            # 随机选一个健康格子
            while True:
                rand_x = np.random.randint(3, self.grid_size[0]-3)
                rand_y = np.random.randint(3, self.grid_size[1]-3)
                if self.grid["burning_state"][rand_x, rand_y]==0:
                    break

            # 创建无人机
            uav = UAV(uav_id, rand_x, rand_y, e=self.uav_energy, d=5)
            # uav = UAV(uav_id, spawn_points[uav_id-1][0], spawn_points[uav_id-1][1], e=300, d=5)
            self.uavs.append(uav)

    def _initialize_rescuers(self):
        """初始化救援人员：从(1,1)出发"""
        self.rescuers = []
        H, W = self.grid_size
        
        # 起点默认在基站附近（基站在 (0, W/2)），人员一开始被基站覆盖，随后走向火场
        start_x, start_y = float(self.rescuer_start[0]), float(self.rescuer_start[1])
        
        # 沿基站一侧铺开一条起始线（垂直于地图内侧方向），而不是全部叠在一个点上。
        # 全叠在一点时，即使目标各不相同，一开始也会走成一列纵队横穿地图，
        # 看起来像"先跑到地图中间再分散"。铺开后是一条并行推进的横队。
        span = max(1.0, float(self.num_rescuers - 1))
        for rescuer_id in range(1, self.num_rescuers + 1):
            k = rescuer_id - 1
            offset_y = k - span / 2.0                 # 沿基站边铺开
            offset_x = (k % 3) * 0.5                  # 轻微错开，避免完全同排
            x = start_x + offset_x
            y = start_y + offset_y
            
            # 确保位置在边界内
            x = max(0.5, min(H - 1.5, x))
            y = max(0.5, min(W - 1.5, y))
            
            rescuer = Rescuer(rescuer_id, float(x), float(y))
            rescuer.status = "idle"  # 初始状态为 idle，等待火情结构感知完成后分配任务
            rescuer.move_counter = 0  # 初始化移动计数器
            self.rescuers.append(rescuer)

    def at_station(self, x, y):
        """(x, y) 是否落在任一补给点的有效范围内（切比雪夫 <= charge_radius）。

        充电判定、奖励里的 on_station 都走这一个函数，避免两边口径不一致。
        """
        r = int(getattr(self, "charge_radius", 1))
        ix, iy = int(round(x)), int(round(y))
        for cx, cy in self.charging_stations:
            if max(abs(ix - int(cx)), abs(iy - int(cy))) <= r:
                return True
        return False

    def nearest_charger(self, uav_xy):
        ux, uy = uav_xy
        best = None; best_d2 = 1e9
        for cx, cy in self.charging_stations:
            d2 = (ux - cx)**2 + (uy - cy)**2
            if d2 < best_d2:
                best_d2 = d2; best = (cx, cy)
        return best  # (cx, cy)
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
        elec_value = np.array([uav.e / 1000.0], dtype=np.float32)
        obs = np.concatenate([
            memory_value,       # 1维
            local_image,        # 9维
            rotation_vec,       # 2维 (单位化)
            avoidance_value,    # 1维
            nearest_vec,         # 2维 (单位化)
            pos_value,           #2
            ammo_value,           #1
            elec_value              #1
        ])

        return obs



    from math import sqrt

    def _compute_spread_speed(self):
        """计算 8 个方向的火势扩散速度（向量化实现）。

        与逐格 Python 循环的原实现数值一致，只是把三重循环换成 numpy——
        任务区域从 50x50 放大到 100x100 后，原实现每次 reset 要跑 76832 次
        math.exp，成为采样瓶颈。
        """
        H, W = self.grid_size
        angle_offset = np.arange(8) * (math.pi * 0.25)
        h = self.grid['height'].astype(np.float64)
        X, Y = np.meshgrid(np.arange(1, H - 1), np.arange(1, W - 1), indexing="ij")
        h_self = h[X, Y]
        for d, (dx, dy) in enumerate(directions_8):
            NX, NY = X + dx, Y + dy
            valid = (NX > 0) & (NX < H - 1) & (NY > 0) & (NY < W - 1)
            h_nb = h[NX.clip(0, H - 1), NY.clip(0, W - 1)]
            height_diff = np.abs(h_self - h_nb)
            height_diff = np.where(height_diff == 0.0, 0.01, height_diff)
            divisor = (1.414 * 100) if d in (1, 3, 5, 7) else 100.0
            sign = np.where(h_self > h_nb, -1.0, 1.0)          # 下坡取负
            slope_factor = np.exp(3.533 * sign * height_diff / divisor)
            wind_correction = math.cos(angle_offset[d] - self.wind_direction)
            speed = (self.R0 * self.ks
                     * math.e ** (0.1783 * self.wind_speed * wind_correction)
                     * slope_factor * self.spread_gain)
            buf = self.grid['spread_speed'][..., d]
            buf[X, Y] = np.where(valid, speed, np.nan)
    def update_burning_state(self, time_step=1.0):
        """更新燃烧状态，只有完全燃烧的格子才会传播火焰（向量化实现）。

        与原逐格实现的一个差别：原实现在同一趟循环里就把烧尽的格子改成 2，
        于是"先处理的格子"和"后处理的格子"看到的火源集合不同（顺序相关）。
        这里统一用**进入本次调用时**的状态作为火源集合，结果与格子遍历顺序无关。
        """
        a = 100.0  # 格子边长 100m
        H, W = self.grid_size
        bs = self.grid['burning_state']
        special = (bs == 2) | (bs == 3)
        full = (bs == 1.0)
        growing = ~special & ~full

        # 完全燃烧的格子计时烧尽；其余重置计时器
        self.burn_time_counter[full] += 1
        self.burn_time_counter[growing] = 0

        # 火源集合取自进入本次调用时的状态
        src = full
        sp = self.grid['spread_speed']
        X, Y = np.meshgrid(np.arange(1, H - 1), np.arange(1, W - 1), indexing="ij")
        sum_R = np.zeros(X.shape, dtype=np.float64)
        sum_R2 = np.zeros(X.shape, dtype=np.float64)
        for d, (dx, dy) in enumerate(directions_8):
            NX, NY = X + dx, Y + dy
            valid = (NX > 0) & (NX < H - 1) & (NY > 0) & (NY < W - 1)
            cx, cy = NX.clip(0, H - 1), NY.clip(0, W - 1)
            contrib = valid & src[cx, cy]
            v = np.nan_to_num(sp[cx, cy, (d + 4) % 8].astype(np.float64))
            if d in (0, 2, 4, 6):
                sum_R += np.where(contrib, v, 0.0)
            else:
                sum_R2 += np.where(contrib, v * v, 0.0)

        term1 = sum_R * time_step / a
        term2 = sum_R2 * time_step / (2 * a ** 2)
        new_inner = np.minimum(1.0, bs[X, Y] + term1 + term2)

        new_burning_state = bs.copy()
        new_burning_state[X, Y] = new_inner
        # 烧尽判定放在传播之后，避免刚烧尽的格子在同一步里被写回 <1 的值
        burnt_out = full & (self.burn_time_counter >= 50)
        mask_update = ~special & ~full
        bs[mask_update] = new_burning_state[mask_update]
        bs[burnt_out] = 2
    def compute_boundary_explore_rate(self, map_info_list=None):
        """
        真·边界：env.grid['burning_state'] 中，火格且八邻域至少一个为0（健康）
        已知边界：在这些边界火格上，map_info 已识别为火/烧尽(>0 或≈2)，
                且存在至少一个邻居在 map_info 中被识别为健康(=0)
        
        参数:
            map_info_list: 所有无人机的map_info列表。如果为None，则使用所有uavs的map_info并取最大值
        返回 (rate, true_cnt, known_cnt)
        """
        M = self.grid["burning_state"]
        H, W = M.shape

        # 如果未提供map_info_list，则使用所有无人机的map_info并取最大值
        if map_info_list is None:
            if len(self.uavs) > 0:
                map_info_list = [uav.map_info for uav in self.uavs if hasattr(uav, 'map_info')]
            else:
                map_info_list = []
        
        # 取所有map_info的最大值，获得整个无人机群的最新认知
        if len(map_info_list) > 0:
            # 使用numpy.maximum.reduce取所有map_info的最大值
            map_info = np.maximum.reduce(map_info_list)
        else:
            # 如果没有map_info，创建一个全-1的数组（表示未知）
            map_info = np.full((H, W), -1.0, dtype=np.float32)

        # 火格（含烧尽与已扑灭）：0=健康, 1/0~1=燃烧, 2=烧尽, 3=已扑灭
        # 为了避免后期灭火后 denominator 过小，引入已扑灭(=3) 作为“历史火边界”参与统计
        is_fire = ((M > 0) & (M <= 1)) | np.isclose(M, 2.0) | np.isclose(M, 3.0)

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
        # 1) 边界火格在 map_info 中被识别为火/烧尽/已扑灭（不是仅非未知）
        known_fire = ((map_info > 0)&(map_info<=1)) | np.isclose(map_info, 2.0) | np.isclose(map_info, 3.0)

        # 2) 至少一个邻居在 map_info 中被识别为健康(=0)
        has_known_healthy_neighbor = np.zeros_like(boundary_true, dtype=bool)
        for dx, dy in dirs8:
            nx = np.clip(np.arange(H)[:, None] + dx, 0, H-1)
            ny = np.clip(np.arange(W)[None, :] + dy, 0, W-1)
            neigh_map = map_info[nx, ny]
            has_known_healthy_neighbor |= (neigh_map == 0)

        # 已知边界 = 真边界 ∩ 已知为火/烧尽 ∩ 邻居已知健康
        known_boundary_mask = boundary_true & known_fire & has_known_healthy_neighbor
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
            return 10
        arr = np.abs(np.array(fire_cells) - np.array([[x, y]])).sum(axis=1)
        return float(arr.min())
    
    def _consume_energy(self, uav, idx):
        """扣除一步能耗并把电量夹紧在 [0, e_max]。

        两处修正：
        1) 原实现 `uav.e -= idx+1` 让同构机队的能耗随索引递增，等于用物理设定
           伪造"分批充电"。现按 energy_cost_mode 统一为常数。
        2) 原实现无下界，200 步后电量会跑到 -500，任何依赖电量的策略/奖励都失效。
        """
        cost = self.energy_cost_per_step(idx)
        uav.e = float(uav.e) - cost
        if uav.e <= 0.0:
            uav.e = 0.0
            uav.depleted = True
        else:
            uav.depleted = False
        return cost

    def energy_cost_per_step(self, uav_index):
        """某架无人机每步的能耗。同构机队 -> 与索引无关。"""
        if getattr(self, "energy_cost_mode", "uniform") == "legacy_index":
            return float(uav_index) + 1.0
        return float(getattr(self, "energy_cost", 2.5))

    def _team_rewards_difference_v2(
        self,
        options
    ):
        """
        按照新的奖励规则计算每架无人机的奖励
        任务编码：0=探索, 1=通信, 2=补给
        """
        import numpy as np
        eps = 1e-8
        
        # ========== 1. 计算全局探索率和覆盖率 ==========
        true_b = max(1, int(getattr(self, "true_boundary", 0)))
        known_b = int(getattr(self, "known_boundary", 0))
        explore_rate = float(known_b) / float(true_b)  # 探索率
        
        # 覆盖率（从环境属性获取）
        coverage_rate = float(getattr(self, "coverage_rate", 0.0))
        
        # ========== 2. 初始化保存上一时间步数据的属性（如果不存在） ==========
        if (not hasattr(self, "_prev_map_info_list")) or (len(getattr(self, "_prev_map_info_list", [])) != len(self.uavs)):
            # 不存在或与当前无人机数量不一致时重新初始化，避免多组合实验时索引越界
            self._prev_map_info_list = [uav.map_info.copy() for uav in self.uavs]
        if not hasattr(self, "_prev_burning_state"):
            self._prev_burning_state = self.grid["burning_state"].copy()
        
        # ========== 3. 计算每架无人机的探索奖励（任务0） ==========
        def _compute_explore_reward(uav_idx, uav):
            """计算探索任务的奖励"""
            # 获取当前和上一时间步的map_info
            prev_map = self._prev_map_info_list[uav_idx]
            curr_map = uav.map_info
            
            # 获取当前和上一时间步的burning_state
            prev_burning = self._prev_burning_state
            curr_burning = self.grid["burning_state"]
            
            # 计算边界格子（火格且至少一个邻居为健康）
            H, W = curr_burning.shape
            dirs8 = [(-1,-1),(-1,0),(-1,1), (0,-1),(0,1), (1,-1),(1,0),(1,1)]
            
            # 当前时间步的边界格子
            is_fire_curr = ((curr_burning > 0) & (curr_burning <= 1)) | np.isclose(curr_burning, 2.0)
            has_healthy_neighbor_curr = np.zeros_like(is_fire_curr, dtype=bool)
            for dx, dy in dirs8:
                nx = np.clip(np.arange(H)[:, None] + dx, 0, H-1)
                ny = np.clip(np.arange(W)[None, :] + dy, 0, W-1)
                has_healthy_neighbor_curr |= (curr_burning[nx, ny] == 0)
            boundary_curr = is_fire_curr & has_healthy_neighbor_curr
            
            # 上一时间步的边界格子
            is_fire_prev = ((prev_burning > 0) & (prev_burning <= 1)) | np.isclose(prev_burning, 2.0)
            has_healthy_neighbor_prev = np.zeros_like(is_fire_prev, dtype=bool)
            for dx, dy in dirs8:
                nx = np.clip(np.arange(H)[:, None] + dx, 0, H-1)
                ny = np.clip(np.arange(W)[None, :] + dy, 0, W-1)
                has_healthy_neighbor_prev |= (prev_burning[nx, ny] == 0)
            boundary_prev = is_fire_prev & has_healthy_neighbor_prev
            
            # a: 本步中是否探索过新发现的边界格子
            # 条件：当前是边界格子，且上一时间步在map_info中未知（<0），当前时间步变为已知（>=0）
            prev_unknown = (prev_map < 0)
            curr_known = (curr_map >= 0) & ((curr_map > 0) | np.isclose(curr_map, 0))
            has_new_boundary = bool(np.any(boundary_curr & prev_unknown & curr_known))
            
            # b: 本步中是否探索过旧格子但火势状态发生了变化
            # 条件：当前是边界格子，上一时间步在map_info中已知（>=0），且burning_state发生了变化
            prev_known = (prev_map >= 0) & ((prev_map > 0) | np.isclose(prev_map, 0))
            fire_state_changed = ~np.isclose(curr_burning, prev_burning, atol=1e-6)
            has_updated_boundary = bool(np.any(boundary_curr & prev_known & fire_state_changed))
            
            # 计算基础奖励（简化版：只要探索过就加固定值，不计算个数）
            reward = 0.0
            if has_new_boundary:
                reward += 0.4
            if has_updated_boundary:
                reward += 0.2
            if reward == 0:
                reward = 0.2
            # 如果全局探索率 < 0.9，额外增加奖励


            return reward
        
        # ========== 4. 计算通信覆盖总人数和通信奖励 ==========
        # 找出所有选择通信任务的UAV索引
        comm_uav_indices = [i for i, opt in enumerate(options) if opt == 1]
        num_comm_uavs = len(comm_uav_indices)
        
        # 计算当前被通信覆盖的总人数（包括基地和所有通信无人机的覆盖）
        R_comm_sq = self.uav_comm_radius ** 2  # 无人机通信半径的平方
        base_comm_radius = 10.0  # 基地通信半径
        base_comm_radius_sq = base_comm_radius ** 2
        base_x, base_y = 0.0, 50.0  # 基地位置
        
        covered_rescuers = set()  # 被覆盖的救援人员ID集合
        
        for rescuer in self.rescuers:
            rx, ry = rescuer.x, rescuer.y
            
            # 1. 检查是否在基地通信覆盖范围内
            dist_to_base_sq = (rx - base_x) ** 2 + (ry - base_y) ** 2
            if dist_to_base_sq <= base_comm_radius_sq:
                covered_rescuers.add(rescuer.rescuer_id)
                continue
            
            # 2. 检查是否在任何通信无人机的通信覆盖范围内
            for uav_idx in comm_uav_indices:
                uav = self.uavs[uav_idx]
                ux, uy = uav.x, uav.y
                dist_sq = (rx - ux) ** 2 + (ry - uy) ** 2
                if dist_sq <= R_comm_sq:
                    covered_rescuers.add(rescuer.rescuer_id)
                    break  # 只要被一个无人机覆盖就算覆盖，不需要继续检查
        
        # 计算被覆盖的总人数
        num_covered = len(covered_rescuers)
        
        # 计算通信总奖励 = 0.06 * 被覆盖人数
        comm_total_reward = 0.06 * float(num_covered)
        
        # 平均分配给所有通信无人机（如果数量>2，第3架及以后仍需要惩罚）
        if num_comm_uavs > 0:
            comm_reward_per_uav = comm_total_reward / float(num_comm_uavs)
        else:
            comm_reward_per_uav = 0.0
        
        
        # ========== 6. 计算所有UAV的奖励 ==========
        rewards = []
        for i, (uav, opt) in enumerate(zip(self.uavs, options)):
            # 获取当前UAV的电量（归一化的，范围0-1）
            # 电量最大值通常是1000，但为了安全起见，使用getattr获取
            uav_e = float(getattr(uav, "e", 1000.0))
            battery_level = uav_e / 1000.0  # 归一化到[0, 1]
            
            if opt == 0:
                # 探索任务
                reward = _compute_explore_reward(i, uav)
            elif opt == 1:
                # 总奖励 = 0.06 * 被覆盖人数，平均分配给所有通信无人机
                reward = comm_reward_per_uav
            elif opt == 2:
                # 补给任务
                reward = 0.1
                # 如果电量大于0.05时选择补给，给予-2的惩罚
                if battery_level > 0.1:
                    reward = -2.0
            else:
                # 其他任务（如opt=3,4等，保留兼容性）
                reward = 0.0
            
            rewards.append(float(reward))
        
        # ========== 7. 更新保存的数据 ==========
        self._prev_map_info_list = [uav.map_info.copy() for uav in self.uavs]
        self._prev_burning_state = self.grid["burning_state"].copy()
        
        return rewards


    def write_local_fire_to_uav_map(self, uav):
        r = int(self.sense_radius)
        x, y = uav.x, uav.y
        H, W = self.grid["burning_state"].shape

        xs = max(0, x - r)
        xe = min(H, x + r + 1)
        ys = max(0, y - r)
        ye = min(W, y + r + 1)

        patch = self.grid["burning_state"][xs:xe, ys:ye]
        # 把"看见的"区域抄到自己的地图；未看的仍保持 -1
        uav.map_info[xs:xe, ys:ye] = patch

    def is_blocked_for_people(self, x: int, y: int) -> bool:
        """
        对于救援人员/人群来说，判断 (x, y) 是否为不可通行格子：
        **只有白色格（burning_state == 0，从未着火）可以走**；
        正在燃烧(0<v<1/1)、烧尽(2)、已扑灭(3) 全部不可通行。

        这与"人员绕火场外圈作业"的设定是自洽的：人员站在火线外侧的白格上向内灭火，
        被扑灭的格子(3) 变成不可通行的隔火带，等外圈连成一圈，里面的火就成了
        `contained`（合围）——按设计本来就不需要再去灭，所以"进不去"不是问题。
        """
        H, W = self.grid_size
        if not (0 <= x < H and 0 <= y < W):
            return True
        return bool(self.grid['burning_state'][x, y] > 0.0)
    
    def _people_blocked_map(self):
        """人员不可通行格（burning_state > 0）的布尔图，按 fire_update_counter 缓存。"""
        key = int(self.fire_update_counter)
        cache = getattr(self, "_blocked_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        # 只有白色格可走：燃烧中 / 烧尽(2) / 已扑灭(3) 一律挡路
        blocked = self.grid['burning_state'] > 0.0
        self._blocked_cache = (key, blocked)
        return blocked

    def _greedy_step_toward(self, sx, sy, gx, gy):
        """朝目标走一步可通行的邻格（BFS 失败时的兜底，避免人员原地卡死）。"""
        blocked = self._people_blocked_map()
        H, W = self.grid_size
        best, best_d = None, None
        for dx, dy in directions_8:
            nx, ny = sx + dx, sy + dy
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            if (nx, ny) != (gx, gy) and blocked[nx, ny]:
                continue
            d = abs(nx - gx) + abs(ny - gy)
            if best_d is None or d < best_d:
                best, best_d = (nx, ny), d
        if best is None or best_d >= abs(sx - gx) + abs(sy - gy):
            return None                      # 没有能让距离变小的可走格，就别乱动
        return best

    def bfs_path_avoid_fire(self, start, goal):
        """
        使用 BFS 在 8 邻域网格上，为救援人员规划一条从 start 到 goal 的路径，
        其中所有中间格子都必须满足 not is_blocked_for_people。
        注意：目标点本身可以是火点（需要扑灭的点），但路径上的中间点不能是火点。
        - start, goal: (sx, sy), (gx, gy) 坐标（整数格）
        - 返回：path = [(x1,y1), (x2,y2), ...]，从 start 下一步开始，到 goal 结束
          如果找不到路径，返回 None
        """
        sx, sy = start
        gx, gy = goal
        H, W = self.grid_size
        
        # 起点和终点检查
        if (sx, sy) == (gx, gy):
            return []
        
        # 注意：目标点本身可以是火点（需要扑灭的点），所以不检查目标点是否为障碍
        # 只检查目标点是否在边界内
        if not (0 <= gx < H and 0 <= gy < W):
            return None
        
        # 阻挡图每步只算一次（is_blocked_for_people 是纯 python，100x100 下
        # 一次 BFS 会调它 50 万次，占到整步耗时的一半）
        blocked = self._people_blocked_map()

        def _search(margin):
            """在 start/goal 包围盒外扩 margin 的范围内搜索；margin=None 表示全图。"""
            if margin is None:
                x0, x1, y0, y1 = 0, H - 1, 0, W - 1
            else:
                x0 = max(0, min(sx, gx) - margin); x1 = min(H - 1, max(sx, gx) + margin)
                y0 = max(0, min(sy, gy) - margin); y1 = min(W - 1, max(sy, gy) + margin)
            q = deque()
            q.append((sx, sy))
            prev = {(sx, sy): None}
            while q:
                x, y = q.popleft()
                if (x, y) == (gx, gy):
                    return prev
                for dx, dy in directions_8:
                    nx, ny = x + dx, y + dy
                    if not (x0 <= nx <= x1 and y0 <= ny <= y1):
                        continue
                    if (nx, ny) in prev:
                        continue
                    # 中间点不能是燃烧格，但目标点允许（目标可能就是要扑灭的火点）
                    if (nx, ny) != (gx, gy) and blocked[nx, ny]:
                        continue
                    prev[(nx, ny)] = (x, y)
                    q.append((nx, ny))
            return prev if (gx, gy) in prev else None

        # 先在小范围里找（快），找不到再全图重试一次——包围盒会把"绕开大火场"的
        # 那条绕行路径切掉，只用小范围会误判为"到不了"。
        prev = _search(16) or _search(None)
        if prev is None:
            return None
        
        # 重建路径
        path = []
        cur = (gx, gy)
        while cur != (sx, sy):
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def write_local_fire_to_rescuer_map(self, rescuer):
        """
        更新救援人员共享的map_info：从八邻域获取火势信息
        救援人员完成一步移动后，获取八邻域的火势信息，填写在环境的rescuers_map_info上
        """
        x, y = rescuer.x, rescuer.y
        H, W = self.grid["burning_state"].shape
        
        # 将位置转为网格索引
        ix = int(round(x))
        iy = int(round(y))
        
        # 八邻域方向（包括自己）
        neighbors_8 = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0), 
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        # 更新八邻域的火势信息到环境的共享地图
        for dx, dy in neighbors_8:
            nx, ny = ix + dx, iy + dy
            if 0 <= nx < H and 0 <= ny < W:
                # 把"看见的"区域抄到环境的共享地图
                self.rescuers_map_info[nx, ny] = self.grid["burning_state"][nx, ny]

    def update_rescuer_fire_structure(self):
        """
        根据救援人员共享的map_info更新环境的B_HEAD、B_FLANK、B_TAIL结构信息
        参考UAV的逻辑，但针对救援人员
        """
        M = self.rescuers_map_info
        H, W = M.shape
        
        # 找到所有燃烧格 (0 < M < 2)
        xs, ys = np.where((M > 0) & (M < 2))
        if xs.size == 0:
            self.rescuers_B_HEAD = []
            self.rescuers_B_FLANK = []
            self.rescuers_B_TAIL = []
            return
        
        # 辅助函数：检查是否有健康邻居
        def has_healthy_neighbor(x, y):
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and M[nx, ny] == 0:
                    return True
            return False
        
        # 1. 计算B_HEAD（火头）：基于x_max回推k格
        k = 1
        x_max = xs.max()
        B_HEAD = []
        for x, y in zip(xs, ys):
            if x >= x_max - k and has_healthy_neighbor(x, y):
                B_HEAD.append((x, y))
        self.rescuers_B_HEAD = B_HEAD
        
        # 2. 计算B_FLANK（侧翼）：基于y_max/y_min回推k格
        y_max = ys.max()
        y_min = ys.min()
        B_FLANK = []
        for x, y in zip(xs, ys):
            # 在左右两条带内
            if (y >= y_max - k or y <= y_min + k):
                # 排除已有的B_HEAD
                if (x, y) not in B_HEAD and has_healthy_neighbor(x, y):
                    B_FLANK.append((x, y))
        self.rescuers_B_FLANK = B_FLANK
        
        # 3. 计算B_TAIL（尾部）：基于x_min前推k格
        x_min = xs.min()
        B_TAIL = []
        for x, y in zip(xs, ys):
            if x <= x_min + k:
                # 排除已有的B_HEAD和B_FLANK
                if (x, y) not in B_HEAD and (x, y) not in B_FLANK and has_healthy_neighbor(x, y):
                    B_TAIL.append((x, y))
        self.rescuers_B_TAIL = B_TAIL

    # ------------------------------------------------------------------
    # 多火点场景：已知火情的连通域聚类，以及"按火点分散"的人员目标分配
    # ------------------------------------------------------------------
    def update_perception_metrics(self):
        """更新面向"多火点搜寻 + 信息回传"场景的指标。

        - explored_rate      : **全地图**已探索比例（不再是火场边界探索率）。
                               多火点场景下"火场探索率"在还没找到火点时恒为 0，
                               无法反映搜寻进度。
        - fire_discovery_rate: 已知燃烧格 / 真实燃烧格，衡量火点找出来了多少。
        - burned_area_rate   : 过火面积比例（燃烧中 + 烧尽 + 已扑灭），越早发现越低。
        - rescuer_known_fire_rate : 人员已掌握的燃烧格 / 真实燃烧格，
                               衡量"无人机有没有把感知地图真的传给人员"。
        """
        H, W = self.grid_size
        total = float(H * W)
        team = self.team_map_info
        if team is None:
            team = self.uavs[0].map_info.copy() if self.uavs else np.full((H, W), -1.0, np.float32)
            for uav in self.uavs[1:]:
                team = np.maximum(team, uav.map_info)
        self.team_map_info = team
        bs = self.grid["burning_state"]
        true_burning = (bs > 0) & (bs < 2)
        n_true = float(true_burning.sum())
        active = self.active_fire_clusters(map_info=team, min_cells=1)
        self.n_contained_clusters = len(
            [c for c in self.known_fire_clusters(map_info=team, min_cells=1)
             if c.get("contained", False)])
        self.n_active_clusters = len(active)

        self.explored_rate = float((team >= 0.0).sum()) / total
        self.burned_area_rate = float((bs > 0).sum()) / total
        known_burning = (team > 0) & (team < 2)
        M = self.rescuers_map_info
        r_known = ((M > 0) & (M < 2)) if M is not None else np.zeros_like(true_burning)

        # 训练时用的口径：分母是全部真实燃烧格（含已合围的）。observation 里的
        # fire_discovery_rate / rescuer_known_fire_rate 继续用它，避免训练/评估不一致。
        self.fire_discovery_rate = (float((known_burning & true_burning).sum()) / n_true
                                    if n_true > 0 else 1.0)
        self.rescuer_known_fire_rate = (float((r_known & true_burning).sum()) / n_true
                                        if n_true > 0 else 1.0)

        # 评估口径：分母只算**未合围**的真实燃烧格。
        # 合围后的火按设计不再派无人机侦查，我们对它的认知必然变旧，
        # 用全部燃烧格做分母会让"发现率"在合围完成后跌到 0——那是设计的结果，
        # 不是策略变差（实测出现过 ext=406 但 discovery=0.000）。
        # 合围判定这里按**真值**做，指标才客观。
        active_burning = self._active_true_burning_mask(bs)
        n_act = float(active_burning.sum())
        self.active_discovery_rate = (float((known_burning & active_burning).sum()) / n_act
                                      if n_act > 0 else 1.0)
        self.active_rescuer_known_rate = (float((r_known & active_burning).sum()) / n_act
                                          if n_act > 0 else 1.0)

    def known_fire_clusters(self, map_info=None, min_cells=1):
        """把**已知**火势图里的燃烧格按 8 连通聚成若干火点。

        原来的 head/flank/tail 结构是按全图 x_max / y_max 推的，地图上只有一个火场
        时没问题；现在初始就有多个分散火点，那套逻辑会把它们当成同一个火场，
        人员于是全部涌向同一个方向。这里改成先聚类，再按火点分配。

        返回 [{"cells", "boundary", "approach", "centroid", "size", "contained"}...]，
        按火点规模从大到小排序。

        `contained=True` 表示该火点**已被合围**：外圈八邻域里既没有已知的健康格(0)、
        也没有未知格，全是烧尽(2)/已扑灭(3)/边界。已扑灭格在 `update_burning_state`
        里是永不改变的（`special` 掩码），等价于隔火带，所以合围后的火不会再扩散——
        既不需要人员再去灭，也不需要无人机再去侦查。
        """
        from scipy import ndimage
        M = self.rescuers_map_info if map_info is None else map_info
        if M is None:
            return []
        if map_info is None:                      # 只缓存默认入参（人员共享图）
            key = (int(self.fire_update_counter), int(min_cells))
            cache = getattr(self, "_cluster_cache", None)
            if cache is not None and cache[0] == key:
                return cache[1]
        burning = (M > 0) & (M < 2)              # 已知且正在燃烧（排除烧尽/已扑灭）
        if not burning.any():
            return []
        lab, n = ndimage.label(burning, structure=np.ones((3, 3), dtype=bool))
        healthy_known = np.isclose(M, 0.0)
        pad = np.pad(healthy_known, 1, mode="constant", constant_values=False)
        has_healthy_nb = np.zeros_like(healthy_known)
        for dx in (0, 1, 2):
            for dy in (0, 1, 2):
                if dx == 1 and dy == 1:
                    continue
                has_healthy_nb |= pad[dx:dx + healthy_known.shape[0],
                                      dy:dy + healthy_known.shape[1]]
        # 接近点：火线**外侧**的白格（已知健康、从未着火）。人员只能站在这里作业。
        # 人员的目标必须落在这里而不是燃烧格上——设计上人员不进火区，围着火区走，
        # 靠近自动灭火。把目标设在燃烧格会让 BFS 的终点始终是"墙"，
        # 火线被扑灭一圈后剩下的燃烧格被围在里面，直接变成不可达。
        flammable = np.isclose(M, 0.0)          # 已知的白格 = 还能烧、也是唯一可通行的
        walkable = flammable
        unknown = M < 0.0
        out = []
        for lid in range(1, n + 1):
            mask = lab == lid
            cnt = int(mask.sum())
            if cnt < min_cells:
                continue
            xs, ys = np.nonzero(mask)
            bx, by = np.nonzero(mask & has_healthy_nb)
            padm = np.pad(mask, 1, mode="constant", constant_values=False)
            near_fire = np.zeros_like(mask)
            for dx in (0, 1, 2):
                for dy in (0, 1, 2):
                    if dx == 1 and dy == 1:
                        continue
                    near_fire |= padm[dx:dx + mask.shape[0], dy:dy + mask.shape[1]]
            ax, ay = np.nonzero(near_fire & walkable)
            ring = near_fire & ~mask
            contained = not bool((ring & (flammable | unknown)).any())
            out.append({
                "cells": list(zip(xs.tolist(), ys.tolist())),
                "boundary": list(zip(bx.tolist(), by.tolist())),
                "approach": list(zip(ax.tolist(), ay.tolist())),
                "centroid": (float(xs.mean()), float(ys.mean())),
                "size": cnt,
                "contained": contained,
            })
        out.sort(key=lambda c: -c["size"])
        if map_info is None:
            self._cluster_cache = ((int(self.fire_update_counter), int(min_cells)), out)
        return out

    def _active_true_burning_mask(self, bs=None):
        """真值意义下"还未被合围"的燃烧格掩码。

        合围 = 该连通火区的外圈八邻域里没有任何可燃格（state 0）。
        已扑灭(3)/烧尽(2) 在 update_burning_state 里永不改变，等价于隔火带。
        """
        from scipy import ndimage
        bs = self.grid["burning_state"] if bs is None else bs
        burning = (bs > 0) & (bs < 2)
        if not burning.any():
            return burning
        lab, n = ndimage.label(burning, structure=np.ones((3, 3), dtype=bool))
        flammable = np.isclose(bs, 0.0)
        pad = np.pad(flammable, 1, mode="constant", constant_values=False)
        has_flammable_nb = np.zeros_like(flammable)
        for dx in (0, 1, 2):
            for dy in (0, 1, 2):
                if dx == 1 and dy == 1:
                    continue
                has_flammable_nb |= pad[dx:dx + flammable.shape[0],
                                        dy:dy + flammable.shape[1]]
        active = np.zeros_like(burning)
        for lid in range(1, n + 1):
            mask = lab == lid
            if (mask & has_flammable_nb).any():
                active |= mask
        return active

    def active_fire_clusters(self, map_info=None, min_cells=1):
        """还需要处理的火点：未被合围的那些。已合围的火不会再扩散，
        既不派人员去灭，也不派无人机去侦查。"""
        return [c for c in self.known_fire_clusters(map_info, min_cells)
                if not c.get("contained", False)]

    def _white_components(self):
        """白格（可通行格）的 8 连通标号图，按步缓存。

        用来判断"这个作业位人员到底走不走得到"。绿色隔火带会把火线外侧的白格切成
        互不连通的口袋：目标格本身是合法的作业位（贴着火），但通往它的唯一走廊
        已经被已扑灭格封死。不做这个过滤的话，人员会一直盯着一个到不了的目标——
        实测 76% 的人步卡在"BFS 无路"。
        """
        key = int(self.fire_update_counter)
        cache = getattr(self, "_white_cc_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        from scipy import ndimage
        white = self.grid["burning_state"] <= 0.0
        lab, _ = ndimage.label(white, structure=np.ones((3, 3), dtype=bool))
        self._white_cc_cache = (key, lab)
        return lab

    def _reachable_ring(self, rescuer, cluster):
        """人员当前所在白格连通域内、属于该火点作业位的格子。"""
        lab = self._white_components()
        sx, sy = int(round(rescuer.x)), int(round(rescuer.y))
        H, W = self.grid_size
        if not (0 <= sx < H and 0 <= sy < W):
            return []
        my = int(lab[sx, sy])
        if my == 0:                       # 自己脚下不是白格（刚被烧到）
            return list(cluster["approach"])
        return [p for p in cluster["approach"] if int(lab[p[0], p[1]]) == my]

    def _perimeter_target(self, rescuer, cluster):
        """给人员在指定火点的外圈上选一个目标格，实现"绕着火场走一圈"。

        两段行为：
          - 还没贴到外圈（距最近外圈格 > 3 格）→ 先去最近的外圈格；
          - 已经贴上了 → 按 `rescuer.perimeter_dir` 的方向沿外圈**角向推进**，
            取角度上"下一个"外圈格。这样人员会绕着火场转而不是在一处来回。
        """
        ring = self._reachable_ring(rescuer, cluster)
        if not ring:
            # 人员对这个火点的认知已经过时了（记忆里的作业位现在全在火里/焦土上），
            # 没有任何当前可站的位置。这时朝火点中心走——路上八邻域会不断刷新
            # 共享火图，走近后就能拿到真实的火线外圈，再转入绕圈作业。
            # 直接返回 None 会让人员干等着，永远等不到新信息。
            cx, cy = cluster["centroid"]
            return (int(round(cx)), int(round(cy)))
        rx, ry = float(rescuer.x), float(rescuer.y)
        cx, cy = cluster["centroid"]
        d_near, near = None, None
        for p in ring:
            d = abs(p[0] - rx) + abs(p[1] - ry)
            if d_near is None or d < d_near:
                d_near, near = d, p
        if d_near is not None and d_near > 3.0:
            return near

        direction = int(getattr(rescuer, "perimeter_dir", 1)) or 1
        my_ang = math.atan2(ry - cy, rx - cx)
        best, best_gap = None, None
        for p in ring:
            gap = (math.atan2(p[1] - cy, p[0] - cx) - my_ang) * direction
            gap = gap % (2.0 * math.pi)          # 归一到 [0, 2pi)
            if gap < 0.15:                        # 太近的当成"就是自己脚下"，跳过
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = p, gap
        return best or near

    def assign_rescuer_targets_by_cluster(self):
        """把人员按名额分散到各个**未合围**火点，并给定绕行方向。

        - 只用 `rescuers_map_info`（自身八邻域 + 无人机上传），不看真值：
          没被无人机告知的火点，人员就是不知道，也不会去。
        - 人员按"离自己最近的火点"排队认领，每个火点名额 ceil(人数/火点数)，
          名额满了去下一个——避免所有人挤同一处。
        - 同一火点内的人员分成顺时针/逆时针两拨，从两侧合围。
        """
        if not self.rescuers:
            return
        clusters = self.active_fire_clusters()
        if not clusters:
            for r in self.rescuers:
                r.target_pos = None
                r.status = "idle"
                r.cluster_anchor = None
            return

        n_r, n_c = len(self.rescuers), len(clusters)
        quota = [n_r // n_c + (1 if i < n_r % n_c else 0) for i in range(n_c)]
        assigned = [0] * n_c
        # 同一火点内先按"当前角度"排好序，再分配均匀铺开的目标角度，
        # 这样一队人是从火场的不同侧面同时贴上去，而不是挤同一个作业位、走同一条路。
        plan_angle = {}
        order = sorted(range(n_r), key=lambda i: min(
            abs(self.rescuers[i].x - c["centroid"][0]) +
            abs(self.rescuers[i].y - c["centroid"][1]) for c in clusters))
        for ri in order:
            r = self.rescuers[ri]
            # 先筛"走得到的火点"，再在其中优先选还有名额的，最后按距离取最近。
            # 顺序不能反：把可达性放在名额之前当排序键时，一个"可达但名额已满"的
            # 火点会压过"名额空着"的火点，10 个人会全挤到同一处（实测过）。
            reach = [ci for ci in range(n_c) if self._reachable_ring(r, clusters[ci])]
            avail = [ci for ci in reach if quota[ci] > 0]
            if not avail:
                # 所有已知火点的名额都满了 → 这个人在基地待命，等新火情
                r.cluster_anchor = None
                r.target_pos = None
                r.status = "idle"
                continue
            ci = min(avail, key=lambda c: abs(r.x - clusters[c]["centroid"][0])
                     + abs(r.y - clusters[c]["centroid"][1]))
            quota[ci] -= 1
            # 同一火点内交替方向，两拨人从两侧绕，合围更快
            r.perimeter_dir = 1 if assigned[ci] % 2 == 0 else -1
            plan_angle.setdefault(ci, []).append(ri)
            assigned[ci] += 1
            r.cluster_anchor = clusters[ci]["centroid"]

        # 名额分完后，把每个火点上的人沿外圈按角度均匀铺开
        for ci, ids in plan_angle.items():
            cl = clusters[ci]
            cx, cy = cl["centroid"]
            m = len(ids)
            for k, ri in enumerate(ids):
                r = self.rescuers[ri]
                ring = self._reachable_ring(r, cl)
                if not ring:
                    r.target_pos = (int(round(cx)), int(round(cy)))
                    r.status = "moving_to_fire_head"
                    continue
                base = math.atan2(r.y - cy, r.x - cx)
                want = base + 2.0 * math.pi * k / m
                tgt = min(ring, key=lambda p: abs(
                    ((math.atan2(p[1] - cy, p[0] - cx) - want + math.pi)
                     % (2.0 * math.pi)) - math.pi))
                r.target_pos = tgt
                r.status = "moving_to_fire_head"
        for r in self.rescuers:
            if getattr(r, "cluster_anchor", None) is None:
                r.target_pos = None
                r.status = "idle"
        self.rescuers_had_target = True

    def _cluster_headcount(self, clusters, exclude=None):
        """当前有多少人锚定在每个火点上（按最近锚点归属）。"""
        cnt = [0] * len(clusters)
        for r in self.rescuers:
            if r is exclude:
                continue
            a = getattr(r, "cluster_anchor", None)
            if a is None:
                continue
            ci = min(range(len(clusters)), key=lambda i: (
                abs(a[0] - clusters[i]["centroid"][0]) +
                abs(a[1] - clusters[i]["centroid"][1])))
            cnt[ci] += 1
        return cnt

    def assign_next_target_for_rescuer_by_cluster(self, rescuer):
        """单个人员推进到外圈的下一个位置；自己负责的火点合围完了才换火点。

        这里也要守住每个火点的人数上限：否则"全局分配留在基地待命的那几个人"
        会在下一步被这个单人函数直接派到唯一已知的那处火，人数上限形同虚设，
        整队又变成排成一列横穿地图。
        """
        clusters = self.active_fire_clusters()
        if not clusters:
            rescuer.target_pos = None
            rescuer.status = "idle"
            rescuer.cluster_anchor = None
            return
        # 不给单个火点设人数上限：试过"未探索完时每处最多 n/3 人、其余在基地待命"，
        # 视觉上确实不再排队横穿地图，但压制能力掉得更多（扑灭 236->166，
        # 火情发现率也跟着掉），因为待命的人什么都没干。
        # 队列感改用"沿外圈按角度铺开目标"来解决，见 assign_rescuer_targets_by_cluster。
        cap = len(self.rescuers)
        head = self._cluster_headcount(clusters, exclude=rescuer)
        anchor = getattr(rescuer, "cluster_anchor", None)
        ref = anchor if anchor is not None else (rescuer.x, rescuer.y)
        ci = min(range(len(clusters)), key=lambda i: (
            abs(ref[0] - clusters[i]["centroid"][0]) +
            abs(ref[1] - clusters[i]["centroid"][1])))
        order = [ci] + [i for i in range(len(clusters)) if i != ci]
        order = [i for i in order if head[i] < cap] or ([ci] if anchor is not None else [])
        for idx in order:
            tgt = self._perimeter_target(rescuer, clusters[idx])
            if tgt is not None:
                rescuer.target_pos = tgt
                rescuer.status = "moving_to_fire_head"
                rescuer.cluster_anchor = clusters[idx]["centroid"]
                return
        rescuer.target_pos = None
        rescuer.status = "idle"

    def compute_true_fire_structures(self, k=1):
        """
        基于环境真实火势 self.grid['burning_state'] 计算三类火场结构（head/flank/tail），
        返回 (B_HEAD, B_FLANK, B_TAIL)，格式与救援人员共享结构一致。
        用于基于真实火势地图直接为救援人员分配目标点。
        """
        M = self.grid["burning_state"]
        H, W = M.shape
        xs, ys = np.where((M > 0) & (M < 2))
        if xs.size == 0:
            return [], [], []

        def has_healthy_neighbor(x, y):
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and M[nx, ny] == 0:
                    return True
            return False

        # head: 基于 x_max 回推 k 格
        x_max = xs.max()
        B_HEAD = []
        for x, y in zip(xs, ys):
            if x >= x_max - k and has_healthy_neighbor(x, y):
                B_HEAD.append((x, y))

        # flank: 基于 y_max / y_min 回推 k 格
        y_max = ys.max()
        y_min = ys.min()
        B_FLANK = []
        for x, y in zip(xs, ys):
            if (y >= y_max - k or y <= y_min + k) and (x, y) not in B_HEAD and has_healthy_neighbor(x, y):
                B_FLANK.append((x, y))

        # tail: 基于 x_min 前推 k 格
        x_min = xs.min()
        B_TAIL = []
        for x, y in zip(xs, ys):
            if x <= x_min + k and (x, y) not in B_HEAD and (x, y) not in B_FLANK and has_healthy_neighbor(x, y):
                B_TAIL.append((x, y))

        return B_HEAD, B_FLANK, B_TAIL

    def filter_rescuer_fire_structure(self):
        """
        筛选救援人员共享的火场结构集合：
        1. 火头集合（B_HEAD）：同一个y坐标，只保留x最大的点
        2. 侧翼集合（B_FLANK）：同一个x坐标，保留y最大的和y最小的点
        3. 尾部集合（B_TAIL）：同一个y坐标，只保留x最小的点
        """
        # 1. 筛选B_HEAD：同一个y坐标，只保留x最大的点
        if len(self.rescuers_B_HEAD) > 0:
            y_groups_head = {}
            for x, y in self.rescuers_B_HEAD:
                if y not in y_groups_head:
                    y_groups_head[y] = []
                y_groups_head[y].append((x, y))
            
            filtered_B_HEAD = []
            for y, points in y_groups_head.items():
                # 找到x最大的点
                max_x_point = max(points, key=lambda p: p[0])
                filtered_B_HEAD.append(max_x_point)
            
            self.rescuers_B_HEAD = filtered_B_HEAD
        
        # 2. 筛选B_FLANK：同一个x坐标，保留y最大的和y最小的点
        if len(self.rescuers_B_FLANK) > 0:
            x_groups_flank = {}
            for x, y in self.rescuers_B_FLANK:
                if x not in x_groups_flank:
                    x_groups_flank[x] = []
                x_groups_flank[x].append((x, y))
            
            filtered_B_FLANK = []
            for x, points in x_groups_flank.items():
                # 找到y最大的点和y最小的点
                max_y_point = max(points, key=lambda p: p[1])
                min_y_point = min(points, key=lambda p: p[1])
                filtered_B_FLANK.append(max_y_point)
                if max_y_point != min_y_point:  # 如果最大和最小不同，才添加最小点
                    filtered_B_FLANK.append(min_y_point)
            
            self.rescuers_B_FLANK = filtered_B_FLANK
        
        # 3. 筛选B_TAIL：同一个y坐标，只保留x最小的点
        if len(self.rescuers_B_TAIL) > 0:
            y_groups_tail = {}
            for x, y in self.rescuers_B_TAIL:
                if y not in y_groups_tail:
                    y_groups_tail[y] = []
                y_groups_tail[y].append((x, y))
            
            filtered_B_TAIL = []
            for y, points in y_groups_tail.items():
                # 找到x最小的点
                min_x_point = min(points, key=lambda p: p[0])
                filtered_B_TAIL.append(min_x_point)
            
            self.rescuers_B_TAIL = filtered_B_TAIL

    def fuse_team_maps_max(self):
        """
        【兼容接口】全局通信：把所有 UAV 的 map_info 融合为一张全局图。
        若想使用“部分可通信”，请调用 fuse_team_maps_partial。
        """
        if not self.uavs:
            return
        maps = [uav.map_info for uav in self.uavs]
        fused = np.maximum.reduce(maps)
        for uav in self.uavs:
            uav.map_info[...] = fused

    def fuse_team_maps_partial(self, radius: float = None):
        """机间火势地图融合。

        `share_all_observations = True`（默认）时全队无条件共享——上层是**集中式**
        规划（一次性给出所有无人机的任务与目标），共享观测才与之自洽。
        置为 False 时退回原来的"距离 <= radius 才交换"的部分可通信模型。
        """
        if not self.uavs:
            return
        if getattr(self, "share_all_observations", True):
            team = self.uavs[0].map_info.copy()
            for uav in self.uavs[1:]:
                np.maximum(team, uav.map_info, out=team)
            for uav in self.uavs:
                uav.map_info[...] = team
            self.team_map_info = team
            return
        if radius is None:
            radius = float(self.communication_radius)
        r2 = radius * radius

        # 复制一份地图，避免就地更新带来的顺序依赖
        maps = [uav.map_info.copy() for uav in self.uavs]
        positions = [(float(u.x), float(u.y)) for u in self.uavs]

        n = len(self.uavs)
        for i in range(n):
            xi, yi = positions[i]
            for j in range(i + 1, n):
                xj, yj = positions[j]
                dx = xi - xj
                dy = yi - yj
                if dx * dx + dy * dy <= r2:
                    fused = np.maximum(maps[i], maps[j])
                    maps[i][...] = fused
                    maps[j][...] = fused

        for uav, m in zip(self.uavs, maps):
            uav.map_info[...] = m

    def update_communication_coverage(self):
        """
        更新救援人员的通信覆盖状态和 AoI（信息时效）
        - 判断每个救援人员是否在至少一个无人机的通信覆盖范围内，或基地(0,0)的通信覆盖范围内
        - 更新每个救援人员的 AoI
        - 计算全局通信特征：coverage_rate 和 max_AoI
        - 信息融合：每个时间步，只要无人机和人群距离小于通信范围，就融合信息
        
        基地位置：(0, 0)，通信半径：10
        """
        if len(self.rescuers) == 0:
            self.coverage_rate = 0.0
            self.max_AoI = 0
            return
        
        R_comm_sq = self.uav_comm_radius ** 2  # 无人机通信半径的平方
        base_comm_radius = float(self.base_comm_radius)  # 基地通信半径
        base_comm_radius_sq = base_comm_radius ** 2
        base_x, base_y = self.base_pos  # 基地位置
        
        num_covered = 0       # 被**无人机**覆盖的人数（口径与奖励一致）
        num_comm_ok = 0       # 有通信的人数（无人机或基站，任一即可）
        max_aoi = 0
        
        # 记录哪些无人机与救援人员进行了通信（用于信息融合和重置time_since_last_comm）
        uavs_in_comm = set()
        
        for rescuer in self.rescuers:
            # 检查是否在任何无人机的通信覆盖范围内，或基地的通信覆盖范围内
            is_covered_now = False
            rx, ry = rescuer.x, rescuer.y
            
            by_uav = False
            # 1. 检查是否在基地通信覆盖范围内
            dist_to_base_sq = (rx - base_x) ** 2 + (ry - base_y) ** 2
            if dist_to_base_sq <= base_comm_radius_sq:
                is_covered_now = True
            # 2. 无人机链路要**无条件**检查，不能放在基站覆盖的 else 分支里。
            #    原实现下人员只要还在基站覆盖圈内，就永远不会与任何无人机建立链路，
            #    于是火势地图传不过去；而人员拿不到火情就不会离开基站 —— 死锁。
            #    是否"被覆盖"和"能不能收到无人机的地图"是两件事。
            for uav_idx, uav in enumerate(self.uavs):
                ux, uy = uav.x, uav.y
                dist_sq = (rx - ux) ** 2 + (ry - uy) ** 2
                if dist_sq <= R_comm_sq:
                    is_covered_now = True
                    by_uav = True
                    uavs_in_comm.add(uav_idx)
                    # 不 break：可能有多架无人机同时覆盖同一个人员
            
            # 更新覆盖状态和 AoI
            if is_covered_now:
                rescuer.is_covered = True
                rescuer.AoI = 0  # 被覆盖时重置 AoI
                num_comm_ok += 1
                if by_uav:
                    num_covered += 1
            else:
                rescuer.is_covered = False
                rescuer.AoI += 1  # 未被覆盖时 AoI 增加
                if rescuer.AoI > max_aoi:
                    max_aoi = rescuer.AoI
        
        # 信息融合：对于所有与救援人员通信的无人机，双向交换map_info（使用max融合）
        for uav_idx in uavs_in_comm:
            uav = self.uavs[uav_idx]
            # 重置该无人机的time_since_last_comm
            uav.time_since_last_comm = 0
            
            # 双向融合：UAV的map_info和救援人员的map_info互相融合（都取最大值）
            if hasattr(uav, "map_info") and uav.map_info is not None:
                # 1. 将UAV的map_info合并到救援人员共享的map_info（使用max融合）
                self.rescuers_map_info = np.maximum(self.rescuers_map_info, uav.map_info)
                # 2. 将救援人员的map_info合并回UAV的map_info（使用max融合）
                uav.map_info = np.maximum(uav.map_info, self.rescuers_map_info)
        
        # 更新全局通信特征
        # coverage_rate 只算**无人机**的覆盖贡献：人员起点就在基站覆盖圈内，
        # 把基站算进来的话随机策略也能拿到 1.000，指标完全没有区分度
        # （上一轮对照里随机臂 coverage=1.000 但人员已知火情 0、从未去灭火）。
        # 真正的"有没有通信"用 comm_coverage_rate，供 blackout 判定使用。
        n_r = len(self.rescuers)
        self.coverage_rate = float(num_covered) / n_r if n_r > 0 else 0.0
        self.comm_coverage_rate = float(num_comm_ok) / n_r if n_r > 0 else 0.0
        # cur_max_AoI 是**当前**的最大信息时效；max_AoI 是历史最大值（只增不减，
        # 仅作诊断用）。奖励和观测必须用当前值：历史最大值一旦被拉高就永久不降，
        # 按它罚分等于给中继机一个无法摆脱的固定负项（实测一回合能罚到 −276）。
        self.cur_max_AoI = int(max_aoi)
        if self.max_AoI < max_aoi:
            self.max_AoI = max_aoi

    def _compute_target_score(self, candidate_point, rescuer_pos, point_type, chosen_points):
        """
        评分函数：计算候选目标点的综合评分
        
        参数:
            candidate_point: (x, y) 候选目标点坐标
            rescuer_pos: (rx, ry) 救援人员当前位置
            point_type: str 点的结构类型 ("head", "flank", "tail", "other")
            chosen_points: list[(x, y)] 已经被选中的目标点列表
        
        返回:
            float: 综合评分（分数越高越好）
        """
        px, py = candidate_point
        rx, ry = rescuer_pos
        
        # 1. 距离项：点离当前人员越近得分越高
        distance = abs(px - rx) + abs(py - ry)  # 曼哈顿距离
        S_dist = -distance / self.target_d0
        
        # 2. 结构项：根据点属于头火、侧翼、尾火等赋予不同权重
        if point_type == "head":
            S_struct = self.target_score_head
        elif point_type == "flank":
            S_struct = self.target_score_flank
        elif point_type == "tail":
            S_struct = self.target_score_tail
        else:
            S_struct = self.target_score_other
        
        # 3. 排斥项：如果该点靠近前面已经被选过的目标点，则施加较大的负奖励
        S_repulse = 0.0
        for chosen_point in chosen_points:
            qx, qy = chosen_point
            dist_to_chosen = abs(px - qx) + abs(py - qy)  # 曼哈顿距离
            if dist_to_chosen <= self.target_repulse_radius:
                # 施加排斥罚分（固定负值）
                S_repulse += self.target_repulse_penalty
        
        # 综合评分
        score = (self.target_w_struct * S_struct + 
                self.target_w_dist * S_dist + 
                self.target_w_repulse * S_repulse)
        
        return score
    
    def assign_rescuer_targets(self):
        """
        【目标分配逻辑的入口函数】
        为救援人员分配目标位置，实现"依次选目标 + 打分 + 空间排斥"的任务分配策略
        
        策略说明：
        1. 将所有救援人员列表打乱随机排序
        2. 按顺序依次为每个人分配目标点
        3. 对每个候选点计算综合评分（距离项 + 结构项 + 排斥项）
        4. 选取得分最高的点作为目标
        5. 保证同一个点最多只被一个人选中
        """
        if len(self.rescuers) == 0:
            return

        if getattr(self, "rescuer_knows_only_shared_map", True):
            return self.assign_rescuer_targets_by_cluster()

        # 使用环境真实火势地图计算火场结构（历史行为，仅复现旧实验时使用）
        fire_head_cells, fire_flank_cells, fire_tail_cells = self.compute_true_fire_structures()
        
        # 检查是否有火场信息
        has_fire_info = len(fire_head_cells) > 0 or len(fire_flank_cells) > 0 or len(fire_tail_cells) > 0
        
        # 如果没有火场信息，不分配目标点（人群会直接往(15, 25)走，但不设置target_pos）
        if not has_fire_info:
            # 清除所有救援人员的目标，让他们直接往(15, 25)走
            for rescuer in self.rescuers:
                rescuer.target_pos = None
                rescuer.status = "idle"
            return
        
        # 过滤掉已熄灭的火点（状态 == 3）
        def filter_valid_fire_cells(cell_list):
            valid = []
            for fx, fy in cell_list:
                if (0 <= fx < self.grid_size[0] and 0 <= fy < self.grid_size[1] and
                    self.grid["burning_state"][fx, fy] > 0 and
                    self.grid["burning_state"][fx, fy] < 3):
                    valid.append((fx, fy))
            return valid
        
        fire_head_cells = filter_valid_fire_cells(fire_head_cells)
        fire_flank_cells = filter_valid_fire_cells(fire_flank_cells)
        fire_tail_cells = filter_valid_fire_cells(fire_tail_cells)
        
        # 预先构建候选点列表：来自当前的火场结构集合，并为每个点标记其结构类型
        candidate_points = []  # [(x, y, type), ...]
        for point in fire_head_cells:
            candidate_points.append((point[0], point[1], "head"))
        for point in fire_flank_cells:
            candidate_points.append((point[0], point[1], "flank"))
        for point in fire_tail_cells:
            candidate_points.append((point[0], point[1], "tail"))
        
        # 如果没有候选点，使用默认目标
        if len(candidate_points) == 0:
            default_target = (15, 25)
            for rescuer in self.rescuers:
                rescuer.target_pos = default_target
                rescuer.status = "moving_to_fire_head"
            return
        
        # 1. 将所有救援人员的列表打乱随机排序
        import random
        rescuers_shuffled = self.rescuers.copy()
        random.shuffle(rescuers_shuffled)
        
        # 2. 按顺序依次为每个人分配目标点
        chosen_points = []  # 记录已经被选中的目标点
        available_candidates = candidate_points.copy()  # 可用候选点列表（会被逐步移除）
        
        for rescuer in rescuers_shuffled:
            rx, ry = rescuer.x, rescuer.y
            
            # 如果可用候选点为空，不分配目标点
            if len(available_candidates) == 0:
                rescuer.target_pos = None
                rescuer.status = "idle"
                continue
            
            # 3. 遍历所有候选点，计算综合评分，选取得分最高的点
            best_score = float('-inf')
            best_point = None
            best_type = None
            
            for candidate in available_candidates:
                px, py, point_type = candidate
                score = self._compute_target_score(
                    candidate_point=(px, py),
                    rescuer_pos=(rx, ry),
                    point_type=point_type,
                    chosen_points=chosen_points
                )
                
                if score > best_score:
                    best_score = score
                    best_point = (px, py)
                    best_type = point_type
            
            # 4. 分配目标点
            if best_point is not None:
                rescuer.target_pos = best_point
                # 根据结构类型设置状态
                if best_type == "head":
                    rescuer.status = "moving_to_fire_head"
                elif best_type == "flank":
                    rescuer.status = "moving_to_side"
                else:
                    rescuer.status = "moving_to_fire_head"
                
                # 5. 将选中的点从候选列表中移除，并添加到已选列表
                # 从available_candidates中移除该点
                available_candidates = [c for c in available_candidates 
                                      if (c[0], c[1]) != best_point]
                chosen_points.append(best_point)
            else:
                # 如果没有找到合适的点，不分配目标点
                rescuer.target_pos = None
                rescuer.status = "idle"

    def assign_next_target_for_rescuer(self, rescuer):
        """
        【目标分配逻辑：单个救援人员重新分配目标】
        为单个救援人员分配下一个目标火点，使用打分制策略
        当当前目标被扑灭后，调用此函数重新分配目标
        
        策略说明：
        1. 构建候选点列表（来自火场结构集合）
        2. 收集其他救援人员已选中的目标点（用于排斥项计算）
        3. 对每个候选点计算综合评分（距离项 + 结构项 + 排斥项）
        4. 选取得分最高的点作为新目标
        """
        if getattr(self, "rescuer_knows_only_shared_map", True):
            return self.assign_next_target_for_rescuer_by_cluster(rescuer)

        # 使用环境真实火势地图计算火场结构（历史行为，仅复现旧实验时使用）
        fire_head_cells, fire_flank_cells, fire_tail_cells = self.compute_true_fire_structures()
        
        # 检查是否有火场信息
        has_fire_info = len(fire_head_cells) > 0 or len(fire_flank_cells) > 0 or len(fire_tail_cells) > 0
        
        # 如果没有火场信息，不分配目标点（人群会直接往(15, 25)走，但不设置target_pos）
        if not has_fire_info:
            rescuer.target_pos = None
            rescuer.status = "idle"
            return
        
        # 过滤掉已熄灭的火点（状态 == 3）
        def filter_valid_fire_cells(cell_list):
            valid = []
            for fx, fy in cell_list:
                if (0 <= fx < self.grid_size[0] and 0 <= fy < self.grid_size[1] and
                    self.grid["burning_state"][fx, fy] > 0 and
                    self.grid["burning_state"][fx, fy] < 3):
                    valid.append((fx, fy))
            return valid
        
        fire_head_cells = filter_valid_fire_cells(fire_head_cells)
        fire_flank_cells = filter_valid_fire_cells(fire_flank_cells)
        fire_tail_cells = filter_valid_fire_cells(fire_tail_cells)
        
        # 预先构建候选点列表：来自当前的火场结构集合，并为每个点标记其结构类型
        candidate_points = []  # [(x, y, type), ...]
        for point in fire_head_cells:
            candidate_points.append((point[0], point[1], "head"))
        for point in fire_flank_cells:
            candidate_points.append((point[0], point[1], "flank"))
        for point in fire_tail_cells:
            candidate_points.append((point[0], point[1], "tail"))
        
        # 如果没有候选点，不分配目标点
        if len(candidate_points) == 0:
            rescuer.target_pos = None
            rescuer.status = "idle"
            return
        
        # 收集其他救援人员已选中的目标点（用于排斥项计算）
        chosen_points = []
        for other_rescuer in self.rescuers:
            if other_rescuer is not rescuer and other_rescuer.target_pos is not None:
                chosen_points.append(other_rescuer.target_pos)
        
        rx, ry = rescuer.x, rescuer.y
        
        # 遍历所有候选点，计算综合评分，选取得分最高的点
        best_score = float('-inf')
        best_point = None
        best_type = None
        
        for candidate in candidate_points:
            px, py, point_type = candidate
            score = self._compute_target_score(
                candidate_point=(px, py),
                rescuer_pos=(rx, ry),
                point_type=point_type,
                chosen_points=chosen_points
            )
            
            if score > best_score:
                best_score = score
                best_point = (px, py)
                best_type = point_type
        
        # 分配目标点
        if best_point is not None:
            rescuer.target_pos = best_point
            # 根据结构类型设置状态
            if best_type == "head":
                rescuer.status = "moving_to_fire_head"
            elif best_type == "flank":
                rescuer.status = "moving_to_side"
            else:
                rescuer.status = "moving_to_fire_head"
        else:
            # 如果没有找到合适的点，不分配目标点
            rescuer.target_pos = None
            rescuer.status = "idle"

    def extinguish_near_rescuers(self):
        """靠近即自动灭火：把每个人员切比雪夫半径 extinguish_radius 内的燃烧格扑灭，
        每人每步最多 extinguish_rate 个（由近到远）。

        原实现只在"人员离**指定目标格**曼哈顿距离 <= 4"时扑灭那一个格子，
        目标一旦被别人灭掉或被围住，人员就白站着。改成按邻域自动作用之后，
        人员沿火线走就会连续产生进展，和"围着火区走、靠近就灭火"的设定一致。
        """
        bs = self.grid["burning_state"]
        H, W = self.grid_size
        R = int(self.extinguish_radius)
        rate = int(self.extinguish_rate)
        n_ext = 0
        for rescuer in getattr(self, "rescuers", []):
            rx, ry = int(round(rescuer.x)), int(round(rescuer.y))
            x0, x1 = max(0, rx - R), min(H, rx + R + 1)
            y0, y1 = max(0, ry - R), min(W, ry + R + 1)
            sub = bs[x0:x1, y0:y1]
            fx, fy = np.nonzero((sub > 0) & (sub < 2))
            if fx.size == 0:
                continue
            gx, gy = fx + x0, fy + y0
            order = np.argsort(np.maximum(np.abs(gx - rx), np.abs(gy - ry)))
            for k in order[:rate]:
                bs[gx[k], gy[k]] = 3.0
                n_ext += 1
        self.n_extinguished_step = n_ext
        return n_ext

    def move_rescuers_one_step(self):
        """人员沿"火场外圈"移动。

        设计（按需求）：人员不进火场，绕着火场边界走，靠近就自动灭火，
        所以目标点永远落在火线**外侧**的可通行格上；`rescuer.perimeter_dir`
        决定绕行方向，人员会沿外圈依次推进而不是在一处来回。

        速度 `rescuer_speed` 是"每步走多少格"（默认 0.5，即两步一格）；
        位置是浮点，按 BFS 路径的下一个整数格做线性插值推进。
        """
        speed = float(getattr(self, "rescuer_speed", 0.5))
        blocked = self._people_blocked_map()
        H, W = self.grid_size
        for rescuer in getattr(self, "rescuers", []):
            if rescuer.target_pos is None or rescuer.status == "idle":
                # 没有已知火情：原地待命，只更新自身八邻域感知
                self.write_local_fire_to_rescuer_map(rescuer)
                continue

            rx, ry = float(rescuer.x), float(rescuer.y)
            sx, sy = int(round(rx)), int(round(ry))
            # 脚下的格子被火烧到了（不再是白格）→ 先撤出去。
            # 优先退到白格；如果被火包住、四周没有白格，允许踩烧尽/已扑灭格撤离——
            # 否则人员会被永久埋在火场疤痕里出不来（实测有 2 人整回合被困）。
            if blocked[sx, sy]:
                bs_arr = self.grid["burning_state"]
                esc, esc_score = None, None
                for allow_burnt in (False, True):
                    for dx, dy in directions_8:
                        nx, ny = sx + dx, sy + dy
                        if not (0 <= nx < H and 0 <= ny < W):
                            continue
                        v = bs_arr[nx, ny]
                        if v > 0.0 and not (allow_burnt and v >= 2.0):
                            continue          # 正常只走白格；紧急撤离时允许踩烧尽/已扑灭
                        x0, x1 = max(0, nx - 1), min(H, nx + 2)
                        y0, y1 = max(0, ny - 1), min(W, ny + 2)
                        score = int(blocked[x0:x1, y0:y1].sum())
                        if esc_score is None or score < esc_score:
                            esc, esc_score = (nx, ny), score
                    if esc is not None:
                        break               # 有白格可退就不用踩火场
                if esc is not None:
                    rescuer.x, rescuer.y = float(esc[0]), float(esc[1])
                    rescuer.cached_path = None
                    rescuer.cached_path_target = None
                    self.write_local_fire_to_rescuer_map(rescuer)
                    continue
            tx, ty = float(rescuer.target_pos[0]), float(rescuer.target_pos[1])
            gx, gy = int(round(tx)), int(round(ty))

            if (sx, sy) == (gx, gy):
                rescuer.cached_path = None
                rescuer.cached_path_target = None
                self.write_local_fire_to_rescuer_map(rescuer)
                continue

            need_replan = (rescuer.cached_path is None
                           or rescuer.cached_path_target != (gx, gy)
                           or not rescuer.cached_path
                           or blocked[rescuer.cached_path[0]])
            if need_replan:
                rescuer.cached_path = self.bfs_path_avoid_fire((sx, sy), (gx, gy))
                rescuer.cached_path_target = (gx, gy)

            if rescuer.cached_path:
                nx, ny = rescuer.cached_path[0]
            else:
                # BFS 失败时走一步贪心可通行邻格，不原地干等
                step = self._greedy_step_toward(sx, sy, gx, gy)
                if step is None:
                    self.write_local_fire_to_rescuer_map(rescuer)
                    continue
                nx, ny = step

            # 朝下一个整数格推进 speed 格
            dx, dy = float(nx) - rx, float(ny) - ry
            d = math.hypot(dx, dy)
            if d <= speed + 1e-9:
                rescuer.x, rescuer.y = float(nx), float(ny)
                if rescuer.cached_path:
                    rescuer.cached_path.pop(0)
            else:
                rescuer.x = rx + dx / d * speed
                rescuer.y = ry + dy / d * speed
            rescuer.x = max(0.5, min(H - 1.5, rescuer.x))
            rescuer.y = max(0.5, min(W - 1.5, rescuer.y))
            self.write_local_fire_to_rescuer_map(rescuer)

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
        for uav in self.uavs:
            uav.just_dropped = False

        pre_state = self.grid["burning_state"].copy()  # 本步动作/蔓延前的状态

        pre_head_set  = set(getattr(self.uavs[0], "B", []) or [])
        pre_flank_set = set(getattr(self.uavs[0], "B_flank", []) or [])
        pre_tail_set  = set(getattr(self.uavs[0], "B_tail", []) or [])

        edge_set = set(getattr(uav, "B", [])) \
         | set(getattr(uav, "B_flank", [])) \
         | set(getattr(uav, "B_tail", []))

        # 1️⃣ 先更新位置+火点（避免无人机重叠）
        # 先收集所有无人机的当前位置（用于碰撞检测）
        occupied_positions = {}
        for idx, u in enumerate(self.uavs):
            pos_key = (int(round(u.x)), int(round(u.y)))
            occupied_positions[pos_key] = idx  # 记录位置和对应的无人机索引
        
        # 先计算所有无人机想要移动到的位置
        planned_moves = []
        for idx, (uav, action) in enumerate(zip(self.uavs, actions)):
            if action < 8:
                dx, dy = [
                    (1,0), (1,1), (0,1), (-1,1), (-1,0),
                    (-1,-1), (0,-1), (1,-1)
                ][action]
                new_x = uav.x + dx
                new_y = uav.y + dy
                
                # 边界检查
                if 0 < new_x < self.grid_size[0]-1 and 0 <= new_y < self.grid_size[1]-1:
                    new_pos = (int(round(new_x)), int(round(new_y)))
                    current_pos = (int(round(uav.x)), int(round(uav.y)))
                    planned_moves.append((idx, current_pos, new_pos, new_x, new_y))
                else:
                    planned_moves.append((idx, None, None, None, None))
            else:
                planned_moves.append((idx, None, None, None, None))
        
        # 检查冲突：如果多个无人机想要移动到同一个位置，只允许索引最小的移动
        target_positions = {}  # {new_pos: [uav_indices]}
        for idx, current_pos, new_pos, new_x, new_y in planned_moves:
            if new_pos is not None:
                if new_pos not in target_positions:
                    target_positions[new_pos] = []
                target_positions[new_pos].append(idx)
        
        # 执行移动（避免重叠）
        # ===== 低层训练所需：每步开始时清空事件标记 =====
        for _u in self.uavs:
            _u.blocked = False
            _u.oob = False
            _u.recharged = False
        for idx, (uav, action) in enumerate(zip(self.uavs, actions)):
            if action < 8:
                dx, dy = [
                    (1,0), (1,1), (0,1), (-1,1), (-1,0),
                    (-1,-1), (0,-1), (1,-1)
                ][action]
                new_x = uav.x + dx
                new_y = uav.y + dy
                
                # 边界检查
                if 0 < new_x < self.grid_size[0]-1 and 0 <= new_y < self.grid_size[1]-1:
                    new_pos = (int(round(new_x)), int(round(new_y)))
                    current_pos = (int(round(uav.x)), int(round(uav.y)))
                    
                    # 碰撞检测：检查目标位置是否已被占用
                    can_move = True
                    
                    # 1. 检查是否与其他无人机的当前位置冲突（排除自己）
                    for other_pos, other_idx in occupied_positions.items():
                        if other_idx != idx and new_pos == other_pos:
                            can_move = False
                            break
                    
                    # 2. 检查是否与其他无人机的计划位置冲突
                    # 如果有多个无人机要移动到同一位置，只允许索引最小的移动
                    if can_move and new_pos in target_positions:
                        uavs_to_same_pos = target_positions[new_pos]
                        if len(uavs_to_same_pos) > 1:
                            # 多个无人机要移动到同一位置，只允许索引最小的移动
                            if idx != min(uavs_to_same_pos):
                                can_move = False
                    
                    if can_move:
                        # 可以移动
                        uav.x = new_x
                        uav.y = new_y
                        self._consume_energy(uav, idx)
                        bx, by = int(uav.x), int(uav.y)
                    else:
                        # 不能移动，原地停留
                        uav.blocked = True
                        self._consume_energy(uav, idx)  # 原地停留也消耗少量电量

                else:
                    # 动作会越界：环境丢弃该动作，记录标记并照常计电
                    uav.oob = True
                    self._consume_energy(uav, idx)
            else:
                #原地停留
                    self._consume_energy(uav, idx)

            # 补给判定放在动作分支之外：原实现只在 action<8 时判定，
            # 导致无人机停在补给点上（action==8）反而无法充电，最终耗尽。
            # 范围判定走 at_station()（3x3 停机坪），避免在站点旁边一格耗尽后永久卡死。
            if self.at_station(uav.x, uav.y):
                uav.d = 5
                # 原来写死 300，是 LEGACY_50x50 配置（uav_energy=300）的遗留值。
                # 现在满电 600，写死 300 等于"充电只充到一半"，电量 580 的机
                # 进站后反而掉到 300。必须按本机的 e_max 充满。
                uav.e = float(getattr(uav, "e_max", 300.0))
                uav.recharged = True
                uav.depleted = False

        # 1.5️⃣ 更新各无人机局部火势地图（用于后续分配救援人员目标）
        for uav in self.uavs:
            self.write_local_fire_to_uav_map(uav)
        # 部分可通信：只在距离不超过 communication_radius 的无人机之间交换火势地图
        self.fuse_team_maps_partial()
        
        # 2️⃣ 更新通信覆盖和 AoI（在无人机移动后）
        self.update_communication_coverage()
        
        # 2.5️⃣ 更新所有无人机的time_since_last_comm（每个时间步+1）
        for uav in self.uavs:
            uav.time_since_last_comm += 1
        
        # 3️⃣ 检查并分配救援人员目标
        # 策略：
        # 1. 如果救援人员没有目标，需要初始分配
        # 2. 如果当前目标是(15, 25)，且有火场信息，立即切换到打分机制分配的目标
        # 3. 如果当前目标不是(15, 25)，检查：
        #    - 目标是否已被扑灭（如果是，重新分配）
        #    - 目标是否已保持5步（如果是，强制重新分配）
        # 这样可以避免目标点反复变化，但允许从默认目标(15, 25)快速切换到火场目标，并每5步自动变换
        
        # 检查是否有火场信息
        if getattr(self, "rescuer_knows_only_shared_map", True):
            # 聚类模式：有没有火情看"人员共享图里有没有火点簇"，
            # head/flank/tail 那套结构在多火点下语义不对，已不再维护。
            _clusters = self.active_fire_clusters()
            has_fire_info = bool(_clusters)
            # 刚拿到火情、还没人有归属火点时，先做一次带名额的全局分配，
            # 否则每人各自挑"离我最近的火点"，会全队挤到同一处。
            if has_fire_info and all(getattr(r, "cluster_anchor", None) is None
                                     for r in self.rescuers):
                self.assign_rescuer_targets_by_cluster()
            elif has_fire_info:
                # 分散作业：两个触发条件
                #   a) 有人负责的火点已经合围/消失（锚点对不上任何活火点）；
                #   b) 每 rebalance_every 步做一次周期性再平衡。
                # 白格限制下人员的可达白连通域会被隔火带切开，单人重选目标时会
                # 逐渐全部漂到"唯一还走得到"的那个火点上（实测 10 人全挤到 1 处），
                # 所以必须周期性地用带名额的全局分配把人手重新摊开。
                cents = [c["centroid"] for c in _clusters]
                stale = any(
                    getattr(r, "cluster_anchor", None) is None
                    or all(abs(r.cluster_anchor[0] - cx) + abs(r.cluster_anchor[1] - cy) > 6.0
                           for cx, cy in cents)
                    for r in self.rescuers)
                period = int(getattr(self, "rebalance_every", 25))
                if stale or (period > 0 and self.fire_update_counter % period == 0):
                    self.assign_rescuer_targets_by_cluster()
        else:
            has_fire_info = (len(self.rescuers_B_HEAD) > 0 or
                             len(self.rescuers_B_FLANK) > 0 or
                             len(self.rescuers_B_TAIL) > 0)
        
        for rescuer in getattr(self, 'rescuers', []):
            # 情况1：救援人员没有目标，且有火场信息，需要初始分配
            if rescuer.target_pos is None and has_fire_info:
                self.assign_next_target_for_rescuer(rescuer)
                rescuer.target_steps = 0  # 重置步数计数器
            # 情况2：救援人员没有目标，但没有火场信息，保持没有目标（会直接往15,25走）
            elif rescuer.target_pos is None and not has_fire_info:
                # 保持没有目标，不分配
                continue
            # 情况3：救援人员有目标（且不是15, 25），检查目标是否已被扑灭或已保持5步
            elif rescuer.status != "idle":
                # 增加目标保持步数
                rescuer.target_steps += 1
                
                # 检查是否已保持5步，如果是则强制重新分配
                if rescuer.target_steps >= 5:
                    self.assign_next_target_for_rescuer(rescuer)
                    rescuer.target_steps = 0  # 重置步数计数器
                    continue
                
                tx, ty = rescuer.target_pos
                # 目标现在是"火线外圈的可通行格"，不再是燃烧格。
                # 到位了（或身边的火已经灭干净了）就沿外圈推进到下一个位置。
                arrived = (abs(rescuer.x - tx) < 0.75 and abs(rescuer.y - ty) < 0.75)
                ix, iy = int(round(rescuer.x)), int(round(rescuer.y))
                R = int(self.extinguish_radius)
                x0, x1 = max(0, ix - R), min(self.grid_size[0], ix + R + 1)
                y0, y1 = max(0, iy - R), min(self.grid_size[1], iy + R + 1)
                sub = self.grid["burning_state"][x0:x1, y0:y1]
                no_fire_nearby = not bool(((sub > 0) & (sub < 2)).any())
                if arrived or no_fire_nearby:
                    self.assign_next_target_for_rescuer(rescuer)
                    rescuer.target_steps = 0
        
        # 4.5️⃣ 地面救援人员移动一步
        self.move_rescuers_one_step()
        
        # 4.6️⃣ 靠近即自动灭火：扑灭人员身边 extinguish_radius 内的燃烧格
        self.extinguish_near_rescuers()

        # 5️⃣ 更新火焰
        if self.fire_update_counter % 10 == 0:
            self.update_burning_state(time_step=1)
            fire_now = int(np.count_nonzero(self.grid["burning_state"] >0))
            self.fire_diff = fire_now - self.fire_cells_prev
            self.fire_cells_prev = fire_now

        self.fire_update_counter += 1
        # 3️⃣ 获取新观测
        observations = []
        all_positions = {u.uav_id: (u.x, u.y) for u in self.uavs}
        for uav in self.uavs:
            obs = self.get_observation(
                uav=uav,
                burning_state=self.grid["burning_state"],
                q_fire=(15,25),
                all_uav_positions=all_positions
            )
            observations.append(obs)

        #计算全局观测增量
        kb_prev = int(getattr(self, "know_boundary_pre", 0))
        # 使用所有无人机的map_info的最大值，获得整个无人机群的最新认知
        self.compute_boundary_explore_rate(map_info_list=None)  # 传入None会自动使用所有uavs的map_info并取最大值
        kb_delta = max(0, int(self.known_boundary) - kb_prev)
        self.update_perception_metrics()

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

        # 低层训练用的是 lowlevel/reward.py 里按任务分解的稠密奖励，环境这份粗奖励
        # 会被直接丢掉；100x100 下它要 ~5ms/步，因此默认关掉。
        if getattr(self, "compute_env_rewards", False):
            rewards = self._team_rewards_difference_v2(options_now)
        else:
            rewards = [0.0] * len(self.uavs)
        self.know_boundary_pre = int(self.known_boundary)
        self.extinguish_pre = extinguish_now
        
        # 5️⃣ 终止条件
        # 已删除所有环境层面的终止条件，只通过训练最大步数(max_steps)控制结束
        # 除非达到训练设定的最大步数，否则episode不会提前结束
        terminated = False
        truncated = False

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

        # 绘制 UAV：根据任务状态选择颜色
        # 通信任务（option == 1）：黑色；其他任务：紫色
        for uav in self.uavs:
            if 0 <= uav.x < self.grid_size[0] and 0 <= uav.y < self.grid_size[1]:
                if hasattr(uav, "option") and uav.option == 1:
                    # 通信任务：黑色
                    color_map[uav.y, uav.x] = [0.0, 0.0, 0.0]
                else:
                    # 其他任务：紫色
                    color_map[uav.y, uav.x] = [0.6, 0.0, 0.8]

        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(12, 12))
            self.img = self.ax.imshow(color_map, origin='lower')
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        else:
            self.img.set_data(color_map)

        # 删除之前所有patches（方框、圆圈、三角形等）和任务文本（不删除信息文本）
        [child.remove() for child in self.ax.patches[:]]
        if hasattr(self, '_task_texts'):
            [txt.remove() for txt in self._task_texts]
        self._task_texts = []
        # for (cx, cy) in self.charging_stations:
        #     rect = plt.Rectangle((cx - 0.5, cy - 0.5), 1, 1,
        #                         linewidth=0, facecolor='deepskyblue', alpha=0.8)
        #     self.ax.add_patch(rect)
            
        # 删除目标点绘制

        # 绘制火场结构（使用人群共享的火场结构 rescuers_B_HEAD等）
        # 火头：黄色边框
        if hasattr(self, "rescuers_B_HEAD") and len(self.rescuers_B_HEAD) > 0:
            for bx, by in self.rescuers_B_HEAD:
                    if 0 <= bx < self.grid_size[0] and 0 <= by < self.grid_size[1]:
                        rect = plt.Rectangle(
                            (bx - 0.5, by - 0.5), 1, 1,
                        linewidth=1.5, edgecolor='yellow', facecolor='none', alpha=0.8
                        )
                        self.ax.add_patch(rect)
            
        # 侧翼：黑色边框
        if hasattr(self, "rescuers_B_FLANK") and len(self.rescuers_B_FLANK) > 0:
            for bx, by in self.rescuers_B_FLANK:
                    if 0 <= bx < self.grid_size[0] and 0 <= by < self.grid_size[1]:
                        rect = plt.Rectangle(
                            (bx - 0.5, by - 0.5), 1, 1,
                        linewidth=1.5, edgecolor='black', facecolor='none', alpha=0.8
                        )
                        self.ax.add_patch(rect)
        
        # 尾部：绿色边框
        if hasattr(self, "rescuers_B_TAIL") and len(self.rescuers_B_TAIL) > 0:
            for bx, by in self.rescuers_B_TAIL:
                    if 0 <= bx < self.grid_size[0] and 0 <= by < self.grid_size[1]:
                        rect = plt.Rectangle(
                            (bx - 0.5, by - 0.5), 1, 1,
                        linewidth=1.5, edgecolor='green', facecolor='none', alpha=0.8
                        )
                        self.ax.add_patch(rect)

        # 绘制基地通信覆盖范围（圆形，位置(0,50)，半径10）
        base_comm_radius = 10.0
        base_circle = plt.Circle((0.0, 50.0), base_comm_radius, 
                               color='green', fill=False, linestyle=':', 
                               linewidth=2, alpha=0.6)
        self.ax.add_patch(base_circle)
        
        # 绘制无人机通信覆盖范围（圆形）
        for uav in self.uavs:
            if 0 <= uav.x < self.grid_size[0] and 0 <= uav.y < self.grid_size[1]:
                circle = plt.Circle((uav.x, uav.y), self.uav_comm_radius, 
                                   color='cyan', fill=False, linestyle='--', 
                                   linewidth=1, alpha=0.5)
                self.ax.add_patch(circle)
        
        # 绘制救援人员：用三角形标记（不记录轨迹）
        for rescuer in getattr(self, 'rescuers', []):
            rx, ry = rescuer.x, rescuer.y
            if 0 <= rx < self.grid_size[0] and 0 <= ry < self.grid_size[1]:
                # 绘制三角形（朝上）
                triangle_size = 0.6
                triangle_points = np.array([
                    [rx, ry + triangle_size],      # 顶点
                    [rx - triangle_size, ry - triangle_size],  # 左下
                    [rx + triangle_size, ry - triangle_size]   # 右下
                ])
                if rescuer.is_covered:
                    triangle = plt.Polygon(triangle_points, closed=True, 
                                         facecolor='cyan', edgecolor='darkcyan', 
                                         linewidth=1.5, alpha=0.8)
                else:
                    triangle = plt.Polygon(triangle_points, closed=True, 
                                         facecolor='orange', edgecolor='darkorange', 
                                         linewidth=1.5, alpha=0.8)
                self.ax.add_patch(triangle)
        
        # 删除救援人员目标点和质心位置绘制
        
        # 显示任务标签（保存引用以便后续删除）
        if not hasattr(self, '_task_texts'):
            self._task_texts = []
        for uav in self.uavs:
            task_names = {0: "Explore", 1: "Comm", 2: "Recharge"}
            task_name = task_names.get(uav.option, "Unknown")
            if 0 <= uav.x < self.grid_size[0] and 0 <= uav.y < self.grid_size[1]:
                task_text = self.ax.text(uav.x + 0.5, uav.y + 0.5, task_name, 
                           fontsize=8, color='white', 
                           bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'),
                           ha='center', va='center')
                self._task_texts.append(task_text)

        if self._wind_text is None:
            self._wind_text = self.ax.text(
                0.96, 0.96, '', transform=self.ax.transAxes,
                ha='right', va='top', fontsize=10, color='white',
                bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2')
            )

        # 获取通信特征
        coverage_rate = getattr(self, 'coverage_rate', 0.0)
        max_aoi = getattr(self, 'max_AoI', 0)
        num_rescuers = len(getattr(self, 'rescuers', []))
        num_covered = sum(1 for r in getattr(self, 'rescuers', []) if r.is_covered) if num_rescuers > 0 else 0

        self._wind_text.set_text(
            f"Wind: {self.wind_level} | Spread: {self.spread_steps} | Shape: {self.init_hw}\n"
            f"Rescuers: {num_covered}/{num_rescuers} covered | Coverage: {coverage_rate:.2%}\n"
            f"Max AoI: {max_aoi} | Step: {self.fire_update_counter}"
        )
        
        self.ax.set_title(f"Human-UAV Cooperative Firefighting | Step: {self.fire_update_counter}", fontsize=14)

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

