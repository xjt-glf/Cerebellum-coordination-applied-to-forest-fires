# -*- coding: utf-8 -*-
"""MAPPO 的 Actor / Centralized Critic。

Actor  : 参数共享，输入 o_i (含任务标签与任务建议) -> 9 个离散动作的 logits
Critic : 置换不变的中心化价值网络，输入 [own_obs, mean(others_obs), global_feat]
         这样无论 2/4/6 架无人机都用同一套参数，不需要按数量重训。

初始化与超参遵循 Yu et al., "The Surprising Effectiveness of PPO in
Cooperative Multi-Agent Games" (MAPPO, NeurIPS 2022) 的推荐做法：
正交初始化、输入层 LayerNorm、价值目标做 running 归一化。
"""
import numpy as np
import torch
import torch.nn as nn


def orth_init(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class MLPBase(nn.Module):
    def __init__(self, in_dim, hidden=256, n_layers=2, use_layernorm=True):
        super().__init__()
        layers = []
        if use_layernorm:
            layers.append(nn.LayerNorm(in_dim))
        d = in_dim
        for _ in range(n_layers):
            layers.append(orth_init(nn.Linear(d, hidden)))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden))
            d = hidden
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    """输出 9 个离散动作的分布。这是最终要交付的"底层模型"。"""

    def __init__(self, obs_dim, n_actions=9, hidden=256, n_layers=2):
        super().__init__()
        self.base = MLPBase(obs_dim, hidden, n_layers)
        self.head = orth_init(nn.Linear(hidden, n_actions), gain=0.01)
        self.n_actions = n_actions

    def forward(self, obs):
        return self.head(self.base(obs))          # logits [B, 9]

    def dist(self, obs, action_mask=None):
        logits = self.forward(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        return torch.distributions.Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs, deterministic=False, action_mask=None):
        d = self.dist(obs, action_mask)
        if deterministic:
            a = d.probs.argmax(dim=-1)
        else:
            a = d.sample()
        return a, d.log_prob(a), d.entropy()

    def evaluate(self, obs, actions, action_mask=None):
        d = self.dist(obs, action_mask)
        return d.log_prob(actions), d.entropy()


class CentralizedCritic(nn.Module):
    """V(s) —— 输入是置换不变的团队状态，与智能体数量无关。"""

    def __init__(self, obs_dim, global_dim, hidden=256, n_layers=2):
        super().__init__()
        self.base = MLPBase(obs_dim * 2 + global_dim, hidden, n_layers)
        self.head = orth_init(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, own_obs, mean_other_obs, global_feat):
        x = torch.cat([own_obs, mean_other_obs, global_feat], dim=-1)
        return self.head(self.base(x)).squeeze(-1)      # [B]


class ValueNorm(nn.Module):
    """价值目标的 running 归一化（MAPPO 中对稳定性影响很大）。"""

    def __init__(self, beta=0.99999, eps=1e-5):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.register_buffer("count", torch.zeros(1))
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1).float()
        bm, bv, bc = x.mean(), x.var(unbiased=False), torch.tensor(
            float(x.numel()), device=x.device)
        w = self.beta ** bc
        new_count = self.count * w + bc
        delta = bm - self.mean
        new_mean = self.mean + delta * bc / new_count.clamp(min=1.0)
        m_a = self.var * (self.count * w)
        m_b = bv * bc
        new_var = (m_a + m_b + delta.pow(2) * (self.count * w) * bc /
                   new_count.clamp(min=1.0)) / new_count.clamp(min=1.0)
        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp(min=self.eps))
        self.count.copy_(new_count)

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var)

    def denormalize(self, x):
        return x * torch.sqrt(self.var) + self.mean


def build_team_inputs(obs_batch, global_batch):
    """把 [B, N, obs_dim] 展开成 critic 需要的 (own, mean_other, global)。

    obs_batch    : Tensor [B, N, obs_dim]
    global_batch : Tensor [B, global_dim]
    返回三个张量，形状分别 [B*N, obs_dim], [B*N, obs_dim], [B*N, global_dim]
    """
    B, N, D = obs_batch.shape
    total = obs_batch.sum(dim=1, keepdim=True)                    # [B,1,D]
    if N > 1:
        mean_other = (total - obs_batch) / float(N - 1)
    else:
        mean_other = torch.zeros_like(obs_batch)
    own = obs_batch.reshape(B * N, D)
    oth = mean_other.reshape(B * N, D)
    glb = global_batch.unsqueeze(1).expand(B, N, global_batch.shape[-1]).reshape(B * N, -1)
    return own, oth, glb
