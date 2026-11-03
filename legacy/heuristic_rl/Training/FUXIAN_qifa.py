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
            # 第二阶段: 巡逻
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

        #补充弹药
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
        else:
        # 安全找索引
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

    # 返回索引
    # print(f"w_k是{w_k}")
    # print(f"相加是{w_k+best_u}")
    # print(f"e_k是{e_k}")
    if e_k==1.0 and np.linalg.norm(w_k+best_u)<=1:
        return 8  # 停下
    else:
    # 安全找索引
        for idx, a in enumerate(action_set):
            if np.allclose(a, best_u):
                return idx

    raise ValueError("Best action not found in action_set!")



def train_madqn(env, num_episodes=10000, gamma=0.99,
                epsilon_start=1.0, epsilon_end=0.05, 
                batch_size=64, target_update_freq=10):
    
    print(f"🖥 当前使用设备: {device}")
    replay_buffer = deque(maxlen=10000000)
    epsilon = epsilon_start
    episode_rewards = []
    q_losses = []
    burning_area_list = []
    while len(replay_buffer) <5000:
        print("启发式经验注入中...")
        obs_list,_ = env.reset()
        done=False
        while not done:
            actions = heuristic_actions(obs_list)
            # print(obs_list)
            next_obs_list, rewards, terminated, truncated, infos = env.step(actions)
            env.render()
            obs_list = next_obs_list
            if terminated or truncated:
                break

   

if __name__ == "__main__":
    # 创建环境
    env = GridEnv()
    
    obs_dim = 18   
    action_dim = 9 
    

    # 训练
    train_madqn(
        env=env,
        num_episodes=10000,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.15,
        batch_size=64,
        target_update_freq=10
    )
