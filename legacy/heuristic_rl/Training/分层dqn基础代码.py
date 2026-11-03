# Minimal runnable demo: 2-option DQN meta-controller learning to switch
# between "Explore" (reward decreases with episode) and "Extinguish" (reward 
# increases with episode). Each decision represents a 5-step option commitment.
# Goal: by around episode ~50, policy should start preferring Extinguish.

import math
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ------------------ Meta-controller (DQN) ------------------

class MetaNet(nn.Module):
    def __init__(self, obs_dim, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)
    def act(self, obs, eps=0.1):
        if random.random() < eps:
            return random.randrange(self.net[-1].out_features)
        with torch.no_grad():
            q = self.forward(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            return int(q.argmax(dim=1).item())

Transition = namedtuple('Transition', 's a r sn done')

class Replay:
    def __init__(self, cap=50000):
        self.buf = deque(maxlen=cap)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch):
        batch = random.sample(self.buf, batch)
        s = torch.as_tensor(np.vstack([t.s for t in batch]), dtype=torch.float32)
        a = torch.as_tensor([t.a for t in batch], dtype=torch.int64).unsqueeze(1)
        r = torch.as_tensor([t.r for t in batch], dtype=torch.float32).unsqueeze(1)
        sn = torch.as_tensor(np.vstack([t.sn for t in batch]), dtype=torch.float32)
        done = torch.as_tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(1)
        return s,a,r,sn,done
    def __len__(self):
        return len(self.buf)

def soft_update(target, online, tau=0.01):
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.copy_(tau*p.data + (1-tau)*tp.data)

# ------------------ Toy "environment" ------------------
# State = [phase], where phase = episode_index / 100.0
# Action 0 = Explore, reward = 1 - phase (decreasing)
# Action 1 = Extinguish, reward = phase (increasing)
# Each episode has T macro-decisions; each decision represents 5-step option.
# Episode ends at T decisions; rewards are per-decision (5-step already abstracted).

class ToyEnv:
    def __init__(self, total_episodes=100, decisions_per_ep=20):
        self.total_episodes = total_episodes
        self.decisions_per_ep = decisions_per_ep
        self.ep = 0
        self.t = 0
        self.phase = 0.0
    def reset(self, ep_idx):
        self.ep = ep_idx
        self.t = 0
        self.phase = ep_idx / float(self.total_episodes)  # in [0,1]
        return np.array([self.phase], dtype=np.float32)
    def step(self, action):
        # rewards per decision (macro 5-step commitment).
        r_explore = 1.0 - self.phase
        r_extinguish = self.phase
        r = r_explore if action == 0 else r_extinguish
        
        self.t += 1
        done = self.t >= self.decisions_per_ep
        # state does not change within episode; small noise to avoid dead target network issues
        sn = np.array([self.phase], dtype=np.float32)
        return sn, r, done, {}

# ------------------ Training loop ------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
obs_dim = 1
n_actions = 2
online = MetaNet(obs_dim, n_actions).to(device)
target = MetaNet(obs_dim, n_actions).to(device)
target.load_state_dict(online.state_dict())
opt = optim.Adam(online.parameters(), lr=1e-3)
replay = Replay(20000)

gamma = 0.95
batch_size = 64
total_episodes = 200
decisions_per_ep = 20
env = ToyEnv(total_episodes, decisions_per_ep)

def eps_by_ep(ep):
    # High exploration early, decays over episodes
    return max(0.05, 0.8 * (1 - ep/total_episodes))

explore_ratio = []
extinguish_ratio = []
mean_q0, mean_q1 = [], []

for ep in range(total_episodes):
    s = env.reset(ep)
    chosen = []
    # Collect transitions
    for d in range(decisions_per_ep):
        eps = eps_by_ep(ep)
        a = online.act(s, eps)
        sn, r, done, _ = env.step(a)
        replay.push(s, a, r, sn, done)
        s = sn
        chosen.append(a)
        
        # Train online network
        if len(replay) >= 500:
            s_b, a_b, r_b, sn_b, done_b = replay.sample(batch_size)
            s_b = s_b.to(device); a_b = a_b.to(device); r_b = r_b.to(device)
            sn_b = sn_b.to(device); done_b = done_b.to(device)
            
            with torch.no_grad():
                q_next = target(sn_b).max(1, keepdim=True)[0]
                y = r_b + gamma * (1 - done_b) * q_next
            q = online(s_b).gather(1, a_b)
            loss = F.mse_loss(q, y)
            opt.zero_grad(); loss.backward(); opt.step()
            soft_update(target, online, tau=0.01)
    
    # log selection ratios per episode
    chosen = np.array(chosen)
    explore_ratio.append((chosen == 0).mean())
    extinguish_ratio.append((chosen == 1).mean())
    with torch.no_grad():
        ph = torch.as_tensor([[ep/total_episodes]], dtype=torch.float32).to(device)
        q_vals = online(ph).cpu().numpy().squeeze()
        if q_vals.ndim == 0:
            q_vals = np.array([q_vals, q_vals])
        mean_q0.append(q_vals[0])
        mean_q1.append(q_vals[1])

# ------------------ Plots ------------------

episodes = np.arange(total_episodes)

plt.figure()
plt.plot(episodes, explore_ratio, label="Explore chosen ratio")
plt.plot(episodes, extinguish_ratio, label="Extinguish chosen ratio")
plt.xlabel("Episode")
plt.ylabel("Action ratio within episode")
plt.title("Policy switches from Explore to Extinguish")
plt.legend()
plt.show()

plt.figure()
plt.plot(episodes, [1 - e/total_episodes for e in episodes], label="Explore reward (theory)")
plt.plot(episodes, [e/total_episodes for e in episodes], label="Extinguish reward (theory)")
plt.xlabel("Episode")
plt.ylabel("Per-decision reward")
plt.title("Reward schedules (cross near ep=50)")
plt.legend()
plt.show()
