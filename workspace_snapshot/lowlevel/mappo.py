# -*- coding: utf-8 -*-
"""MAPPO 训练器：rollout buffer + GAE + PPO clip 更新。

约定的张量形状（T = rollout 长度, E = 并行环境数, N = 无人机数）
    obs   [T+1, E, N, obs_dim]
    glb   [T+1, E, global_dim]
    act   [T,   E, N]
    logp  [T,   E, N]
    val   [T+1, E, N]
    rew   [T,   E, N]
    done  [T,   E]            (episode 边界，全队共享)
"""
import numpy as np
import torch
import torch.nn as nn

from .models import Actor, CentralizedCritic, ValueNorm, build_team_inputs


class Config:
    # 采样
    rollout = 200
    n_envs = 8
    total_env_steps = 2_000_000
    # 优化
    lr = 5e-4
    critic_lr = 5e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip = 0.2
    ppo_epochs = 5
    n_minibatch = 1
    entropy_coef = 0.01
    value_coef = 1.0
    max_grad_norm = 10.0
    huber_delta = 10.0
    use_value_norm = True
    use_clipped_value_loss = True
    # 网络
    hidden = 256
    n_layers = 2
    # 其它
    seed = 0
    device = "auto"
    log_interval = 5
    save_interval = 50


def resolve_device(name="auto"):
    if name not in ("auto", None):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RolloutBuffer:
    def __init__(self, T, E, N, obs_dim, global_dim, device):
        self.T, self.E, self.N = T, E, N
        self.obs = torch.zeros(T + 1, E, N, obs_dim, device=device)
        self.glb = torch.zeros(T + 1, E, global_dim, device=device)
        self.act = torch.zeros(T, E, N, dtype=torch.long, device=device)
        self.logp = torch.zeros(T, E, N, device=device)
        self.val = torch.zeros(T + 1, E, N, device=device)
        self.rew = torch.zeros(T, E, N, device=device)
        self.done = torch.zeros(T + 1, E, device=device)
        self.adv = torch.zeros(T, E, N, device=device)
        self.ret = torch.zeros(T, E, N, device=device)
        self.step = 0

    def start(self, obs, glb):
        self.obs[0].copy_(obs)
        self.glb[0].copy_(glb)
        self.done[0].zero_()
        self.step = 0

    def add(self, obs, glb, act, logp, val, rew, done):
        t = self.step
        self.act[t].copy_(act)
        self.logp[t].copy_(logp)
        self.val[t].copy_(val)
        self.rew[t].copy_(rew)
        self.obs[t + 1].copy_(obs)
        self.glb[t + 1].copy_(glb)
        self.done[t + 1].copy_(done)
        self.step += 1

    def compute_gae(self, last_val, gamma, lam, value_norm=None):
        self.val[self.T].copy_(last_val)
        gae = torch.zeros(self.E, self.N, device=self.val.device)
        for t in reversed(range(self.T)):
            not_done = (1.0 - self.done[t + 1]).unsqueeze(-1)     # [E,1]
            v_t = self.val[t]
            v_next = self.val[t + 1]
            if value_norm is not None:
                v_t = value_norm.denormalize(v_t)
                v_next = value_norm.denormalize(v_next)
            delta = self.rew[t] + gamma * v_next * not_done - v_t
            gae = delta + gamma * lam * not_done * gae
            self.adv[t].copy_(gae)
            self.ret[t].copy_(gae + v_t)

    def flat(self):
        T, E, N = self.T, self.E, self.N
        D = self.obs.shape[-1]
        G = self.glb.shape[-1]
        obs = self.obs[:T].reshape(T * E, N, D)
        glb = self.glb[:T].reshape(T * E, G)
        return (obs, glb,
                self.act[:T].reshape(-1),
                self.logp[:T].reshape(-1),
                self.adv[:T].reshape(-1),
                self.ret[:T].reshape(-1))


class MAPPO:
    def __init__(self, obs_dim, n_actions, global_dim, cfg: Config = None):
        self.cfg = cfg or Config()
        self.device = resolve_device(self.cfg.device)
        torch.manual_seed(self.cfg.seed)
        self.actor = Actor(obs_dim, n_actions, self.cfg.hidden, self.cfg.n_layers).to(self.device)
        self.critic = CentralizedCritic(obs_dim, global_dim,
                                       self.cfg.hidden, self.cfg.n_layers).to(self.device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr, eps=1e-5)
        self.value_norm = ValueNorm().to(self.device) if self.cfg.use_value_norm else None
        self.obs_dim, self.n_actions, self.global_dim = obs_dim, n_actions, global_dim

    # ------------------------------------------------------------------ 交互
    @torch.no_grad()
    def act(self, obs, glb, deterministic=False):
        """obs [E,N,D], glb [E,G] -> act [E,N], logp [E,N], val [E,N]"""
        E, N, D = obs.shape
        a, lp, _ = self.actor.act(obs.reshape(E * N, D), deterministic=deterministic)
        own, oth, g = build_team_inputs(obs, glb)
        v = self.critic(own, oth, g)
        return a.view(E, N), lp.view(E, N), v.view(E, N)

    # ------------------------------------------------------------------ 更新
    def update(self, buf: RolloutBuffer):
        cfg = self.cfg
        obs, glb, act, logp_old, adv, ret = buf.flat()
        B = obs.shape[0]
        N = obs.shape[1]

        if self.value_norm is not None:
            self.value_norm.update(ret)
            ret_n = self.value_norm.normalize(ret)
        else:
            ret_n = ret

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx_all = np.arange(B)
        mb = max(1, B // cfg.n_minibatch)
        stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0, "n": 0}

        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(idx_all)
            for s in range(0, B, mb):
                sel = torch.as_tensor(idx_all[s:s + mb], device=self.device)
                o = obs[sel]                       # [b,N,D]
                g = glb[sel]                       # [b,G]
                b = o.shape[0]
                flat_sel = (sel.unsqueeze(-1) * N + torch.arange(N, device=self.device)).reshape(-1)

                # ---- actor ----
                lp, ent = self.actor.evaluate(o.reshape(b * N, -1), act[flat_sel])
                ratio = torch.exp(lp - logp_old[flat_sel])
                a_mb = adv[flat_sel]
                s1 = ratio * a_mb
                s2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * a_mb
                pg_loss = -torch.min(s1, s2).mean()
                ent_loss = ent.mean()
                loss_a = pg_loss - cfg.entropy_coef * ent_loss
                self.opt_a.zero_grad(set_to_none=True)
                loss_a.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.opt_a.step()

                # ---- critic ----
                own, oth, gg = build_team_inputs(o, g)
                v_pred = self.critic(own, oth, gg)
                v_targ = ret_n[flat_sel]
                if cfg.use_clipped_value_loss:
                    v_old = buf.val[:buf.T].reshape(-1)[flat_sel]
                    if self.value_norm is not None:
                        v_old = v_old.detach()
                    v_clip = v_old + torch.clamp(v_pred - v_old, -cfg.clip, cfg.clip)
                    l1 = _huber(v_pred - v_targ, cfg.huber_delta)
                    l2 = _huber(v_clip - v_targ, cfg.huber_delta)
                    v_loss = torch.max(l1, l2).mean()
                else:
                    v_loss = _huber(v_pred - v_targ, cfg.huber_delta).mean()
                loss_c = cfg.value_coef * v_loss
                self.opt_c.zero_grad(set_to_none=True)
                loss_c.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.opt_c.step()

                with torch.no_grad():
                    stats["pg"] += float(pg_loss)
                    stats["v"] += float(v_loss)
                    stats["ent"] += float(ent_loss)
                    stats["kl"] += float((logp_old[flat_sel] - lp).mean())
                    stats["clipfrac"] += float(((ratio - 1.0).abs() > cfg.clip).float().mean())
                    stats["n"] += 1

        n = max(1, stats.pop("n"))
        return {k: v / n for k, v in stats.items()}

    # ------------------------------------------------------------------ 存取
    def state_dict(self):
        d = {"actor": self.actor.state_dict(),
             "critic": self.critic.state_dict(),
             "opt_a": self.opt_a.state_dict(),
             "opt_c": self.opt_c.state_dict(),
             "obs_dim": self.obs_dim, "n_actions": self.n_actions,
             "global_dim": self.global_dim,
             "hidden": self.cfg.hidden, "n_layers": self.cfg.n_layers}
        if self.value_norm is not None:
            d["value_norm"] = self.value_norm.state_dict()
        return d

    def load_state_dict(self, d, strict=True):
        self.actor.load_state_dict(d["actor"], strict=strict)
        self.critic.load_state_dict(d["critic"], strict=strict)
        if "opt_a" in d:
            try:
                self.opt_a.load_state_dict(d["opt_a"])
                self.opt_c.load_state_dict(d["opt_c"])
            except Exception:
                pass
        if self.value_norm is not None and "value_norm" in d:
            self.value_norm.load_state_dict(d["value_norm"])


def _huber(x, delta):
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * x.pow(2), delta * (ax - 0.5 * delta))
