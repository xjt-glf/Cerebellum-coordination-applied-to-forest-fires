# -*- coding: utf-8 -*-
"""lowlevel: 任务条件低层强化学习策略（输入 状态+任务指令 → 输出 9 个离散动作）。

上层集中式规划一次给出全队指令（TeamPlan），每架无人机一个共享参数的策略。
"""

from .task_spec import (
    UavOrder, TeamPlan,
    TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE, N_TASKS,
    SENSE_EXPLORE, SENSE_PATROL, N_SENSE_MODES,
    TASK_NAMES, MODE_NAMES,
)
from .obs_builder import OBS_DIM, OBS_SLICES, build_obs
from .reward import RewardConfig, LowLevelReward
from .env_wrapper import TaskConditionedEnv, N_ACTIONS

__all__ = [
    "UavOrder", "TeamPlan",
    "TASK_FIRE_SENSING", "TASK_COMM_RELAY", "TASK_RECHARGE", "N_TASKS",
    "SENSE_EXPLORE", "SENSE_PATROL", "N_SENSE_MODES",
    "TASK_NAMES", "MODE_NAMES",
    "OBS_DIM", "OBS_SLICES", "build_obs",
    "RewardConfig", "LowLevelReward",
    "TaskConditionedEnv", "N_ACTIONS",
]
