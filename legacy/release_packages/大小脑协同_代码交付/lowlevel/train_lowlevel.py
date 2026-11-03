# -*- coding: utf-8 -*-
"""目标条件 PPO 训练入口（H200 计划 §6/§11）。

用法（由 overnight.py 调度，也可单独运行）：
    python lowlevel/train_lowlevel.py \
        --trial-id P1_s1101 --gpu 0 --reward P1 --roles randomized \
        --init actor_only --init-actor models/actor_selected.pt \
        --seed 1101 --total-steps 30000000

初始化路径（计划 §6）：
- actor_only   : 只加载旧 Actor（384 宽 2 层），值网络/Adam/ValueNorm 全部新建，
                 前 5 个 update 只校准值网络；
- random       : 随机初始化 Actor（重训 R1）；
- bc           : 先行为克隆热身（教师数据见 teacher.py），再 PPO（P3/R2）；
- resume_exact : 同一 trial 的完整恢复（模型+优化器+ValueNorm+步数+RNG+课程）。
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lowlevel.models import Actor, CentralizedCritic, ValueNorm, build_team_inputs
from lowlevel.mappo import RolloutBuffer, resolve_device
from lowlevel.vec_env import SubprocVecEnv, DummyVecEnv
from lowlevel.reward import RewardConfig
from lowlevel.goal_env import GoalEnv

CONTRACT = "goal_conditioned_v1"       # 采样/奖励/目标管理契约版本


# ---------------------------------------------------------------- 环境
def make_env_fn(seed, args, steps_offset=0):
    kw = dict(
        reward_version=args.reward,
        role_mode=args.roles,
        stage=args.start_stage,
        stage_steps={"S1": args.stage_s1_steps, "S2": args.stage_s2_steps},
        steps_offset=steps_offset,
        num_uavs=args.num_uavs, max_steps=args.max_steps,
        spread_steps=40, grid_size=(args.grid, args.grid),
        n_fire_seeds=args.fire_seeds, spread_gain=1.0,
        num_rescuers=args.num_rescuers, seed=seed,
        replan_every=args.replan_every,
    )

    def _f():
        return GoalEnv(**kw)
    return _f


# ---------------------------------------------------------------- PPO
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.ent0 = args.entropy_coef

        # resume_exact 时先取已训练步数，用于环境课程阶段的偏移
        steps_offset = 0
        if args.init == "resume_exact":
            # RNG 状态（numpy/torch）需要 weights_only=False；文件来自本 trial 自己的保存
            ck = torch.load(args.init_resume_ckpt, map_location="cpu",
                            weights_only=False)
            steps_offset = int(ck.get("env_steps", 0))

        self.vec = (DummyVecEnv([make_env_fn(args.seed * 1000 + i, args)
                                 for i in range(args.n_envs)])
                    if args.dummy_vec else
                    SubprocVecEnv([make_env_fn(args.seed * 1000 + i, args, steps_offset)
                                   for i in range(args.n_envs)],
                                  envs_per_worker=args.envs_per_worker))
        obs_dim, n_act, glb_dim = self.vec.obs_dim, self.vec.n_actions, self.vec.global_dim

        self.actor = Actor(obs_dim, n_act, hidden=args.hidden, n_layers=args.n_layers).to(self.device)
        self.critic = CentralizedCritic(obs_dim, glb_dim, hidden=args.hidden,
                                        n_layers=args.n_layers).to(self.device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, eps=1e-5)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)
        self.value_norm = ValueNorm().to(self.device)
        self.obs_dim, self.n_actions, self.global_dim = obs_dim, n_act, glb_dim

        self.env_steps = 0
        self.updates = 0
        self.warmup_left = args.critic_warmup
        self._init_weights()

    def _init_weights(self):
        a = self.args
        if a.init == "actor_only":
            sd = torch.load(a.init_actor, map_location="cpu")
            if isinstance(sd, dict) and "actor" in sd:
                sd = sd["actor"]
            hidden = int(sd["head.weight"].shape[1])
            assert hidden == a.hidden, f"init actor hidden={hidden} != cfg {a.hidden}"
            self.actor.load_state_dict(sd)
            print(f"[init] actor_only <- {a.init_actor} (hidden={hidden})")
        elif a.init == "random":
            print("[init] random actor")
        elif a.init == "resume_exact":
            self._resume_exact()
        elif a.init == "bc":
            print("[init] bc warmstart (deferred to pretrain loop)")

    # ------------------------------------------------------------ BC 热身
    def bc_warmstart(self, data_path, epochs_min=3, epochs_max=10, mb=4096, lr=3e-4):
        d = np.load(data_path)
        X, Y = torch.as_tensor(d["obs"]), torch.as_tensor(d["act"]).long()
        n = X.shape[0]
        idx = torch.randperm(n)
        n_dev = max(1000, int(0.1 * n))
        Xd, Yd = X[idx[:n_dev]].to(self.device), Y[idx[:n_dev]].to(self.device)
        Xt, Yt = X[idx[n_dev:]].to(self.device), Y[idx[n_dev:]].to(self.device)
        opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        best, best_state, hist = 1e18, None, []
        for ep in range(epochs_max):
            self.actor.train()
            perm = torch.randperm(Xt.shape[0], device=self.device)
            tot = 0.0
            for s in range(0, Xt.shape[0], mb):
                sel = perm[s:s + mb]
                logits = self.actor(Xt[sel])
                loss = torch.nn.functional.cross_entropy(logits, Yt[sel])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
                opt.step()
                tot += float(loss) * sel.numel()
            self.actor.eval()
            with torch.no_grad():
                dev = float(torch.nn.functional.cross_entropy(
                    self.actor(Xd), Yd))
            hist.append(dev)
            if dev < best:
                best, best_state = dev, {k: v.clone() for k, v in
                                         self.actor.state_dict().items()}
            print(f"[bc] epoch {ep + 1}: train {tot / Xt.shape[0]:.4f} dev {dev:.4f}")
            if ep + 1 >= epochs_min and dev > best * 1.02:
                break
        if best_state is not None:
            self.actor.load_state_dict(best_state)
        return dict(path=data_path, epochs=len(hist), dev_loss=best, history=hist)

    # ------------------------------------------------------------ 交互
    @torch.no_grad()
    def act(self, obs, glb):
        E, N, D = obs.shape
        a, lp, _ = self.actor.act(obs.reshape(E * N, D))
        own, oth, g = build_team_inputs(obs, glb)
        v = self.critic(own, oth, g)
        return a.view(E, N), lp.view(E, N), v.view(E, N)

    def update(self, buf):
        a = self.args
        obs, glb, act, logp_old, adv, ret = buf.flat()
        B, N = obs.shape[0], obs.shape[1]
        if self.value_norm is not None:
            self.value_norm.update(ret)
            ret_n = self.value_norm.normalize(ret)
        else:
            ret_n = ret
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx_all = np.arange(B)
        mb_size = max(1, min(B, a.target_minibatch // N))
        stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0, "gnorm": 0.0, "n": 0}
        for _ in range(a.ppo_epochs):
            np.random.shuffle(idx_all)
            for s in range(0, B, mb_size):
                sel = torch.as_tensor(idx_all[s:s + mb_size], device=self.device)
                o, g = obs[sel], glb[sel]
                b = o.shape[0]
                flat_sel = (sel.unsqueeze(-1) * N + torch.arange(N, device=self.device)).reshape(-1)
                lp, ent = self.actor.evaluate(o.reshape(b * N, -1), act[flat_sel])
                ratio = torch.exp(lp - logp_old[flat_sel])
                a_mb = adv[flat_sel]
                s1 = ratio * a_mb
                s2 = torch.clamp(ratio, 1.0 - a.clip, 1.0 + a.clip) * a_mb
                pg_loss = -torch.min(s1, s2).mean()
                # 值网络热身期只训 critic（Actor 冻结）
                if self.warmup_left <= 0:
                    loss_a = pg_loss - a.entropy_coef * ent.mean()
                    self.opt_a.zero_grad(set_to_none=True)
                    loss_a.backward()
                    gn = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), a.max_grad_norm)
                    self.opt_a.step()
                else:
                    gn = torch.tensor(0.0)
                own, oth, gg = build_team_inputs(o, g)
                v_pred = self.critic(own, oth, gg)
                v_targ = ret_n[flat_sel]
                v_old = buf.val[:buf.T].reshape(-1)[flat_sel].detach()
                v_clip = v_old + torch.clamp(v_pred - v_old, -a.clip, a.clip)
                hub = lambda x: torch.where(x.abs() <= 10.0, 0.5 * x.pow(2),
                                            10.0 * (x.abs() - 5.0))
                v_loss = torch.max(hub(v_pred - v_targ), hub(v_clip - v_targ)).mean()
                self.opt_c.zero_grad(set_to_none=True)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), a.max_grad_norm)
                self.opt_c.step()
                with torch.no_grad():
                    stats["pg"] += float(pg_loss)
                    stats["v"] += float(v_loss)
                    stats["ent"] += float(ent.mean())
                    stats["kl"] += float((logp_old[flat_sel] - lp).mean())
                    stats["clipfrac"] += float(((ratio - 1.0).abs() > a.clip).float().mean())
                    stats["gnorm"] += float(gn)
                    stats["n"] += 1
        n = max(1, stats.pop("n"))
        if self.warmup_left > 0:
            self.warmup_left -= 1
        return {k: v / n for k, v in stats.items()}

    # ------------------------------------------------------------ 存取
    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    opt_a=self.opt_a.state_dict(), opt_c=self.opt_c.state_dict(),
                    value_norm=self.value_norm.state_dict(),
                    env_steps=self.env_steps, updates=self.updates,
                    warmup_left=self.warmup_left,
                    contract=CONTRACT,
                    torch_rng=torch.get_rng_state(), np_rng=np.random.get_state())

    def save_ckpt(self, path):
        tmp = path + ".tmp"
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)

    def _resume_exact(self):
        p = self.args.init_resume_ckpt
        d = torch.load(p, map_location=self.device, weights_only=False)
        assert d.get("contract") == CONTRACT, f"contract mismatch: {d.get('contract')}"
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.opt_a.load_state_dict(d["opt_a"])
        self.opt_c.load_state_dict(d["opt_c"])
        self.value_norm.load_state_dict(d["value_norm"])
        self.env_steps = int(d["env_steps"])
        self.updates = int(d["updates"])
        self.warmup_left = int(d.get("warmup_left", 0))
        torch.set_rng_state(d["torch_rng"].cpu())
        np.random.set_state(d["np_rng"])
        print(f"[resume_exact] {p} @ env_steps={self.env_steps}")

    def save_actor(self, path):
        tmp = path + ".tmp"
        torch.save(self.actor.state_dict(), tmp)
        os.replace(tmp, path)


# ---------------------------------------------------------------- 主循环
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-id", required=True)
    ap.add_argument("--run-root", default=os.path.join(_HERE, "runs"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--reward", default="P1", choices=["P1", "P2", "P1R", "original"])
    ap.add_argument("--roles", default="randomized", choices=["randomized", "legacy_fixed"])
    ap.add_argument("--init", default="actor_only",
                    choices=["actor_only", "random", "bc", "resume_exact"])
    ap.add_argument("--init-actor", default=os.path.join(_ROOT, "models", "actor_selected.pt"))
    ap.add_argument("--init-resume-ckpt", default=None)
    ap.add_argument("--bc-data", default=None)
    ap.add_argument("--bc-epochs-min", type=int, default=3)
    ap.add_argument("--bc-epochs-max", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1101)
    ap.add_argument("--total-steps", type=int, default=30_000_000)
    ap.add_argument("--num-uavs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--grid", type=int, default=100)
    ap.add_argument("--fire-seeds", type=int, default=3)
    ap.add_argument("--num-rescuers", type=int, default=10)
    ap.add_argument("--replan-every", type=int, default=20)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--envs-per-worker", type=int, default=2)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--actor-lr", type=float, default=1e-4)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--entropy-coef", type=float, default=0.02)
    ap.add_argument("--entropy-final", type=float, default=0.01)
    ap.add_argument("--entropy-anneal-steps", type=float, default=10e6)
    ap.add_argument("--max-grad-norm", type=float, default=5.0)
    ap.add_argument("--target-minibatch", type=int, default=4096)
    ap.add_argument("--critic-warmup", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--stage-s1-steps", type=float, default=2e6)
    ap.add_argument("--stage-s2-steps", type=float, default=5e6)
    ap.add_argument("--start-stage", default="S1")
    ap.add_argument("--save-every-sec", type=float, default=600)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--dummy-vec", action="store_true")
    args = ap.parse_args()

    out = os.path.join(args.run_root, args.trial_id)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(os.path.join(out, "contract.txt"), "w") as f:
        f.write(CONTRACT + "\n")

    tr = Trainer(args)
    if args.init == "bc":
        path = args.bc_data or os.path.join(args.run_root, "teacher_bc.npz")
        if not os.path.exists(path):
            from lowlevel.teacher import generate_teacher_data
            generate_teacher_data(path, target_transitions=100000,
                                  workers=max(2, args.n_envs // 2))
        info = tr.bc_warmstart(path, args.bc_epochs_min, args.bc_epochs_max)
        with open(os.path.join(out, "bc_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        tr.save_ckpt(os.path.join(out, "ckpt_after_bc.pt"))

    buf = RolloutBuffer(args.rollout, tr.vec.n_envs, tr.vec.num_uavs,
                        tr.vec.obs_dim, tr.vec.global_dim, tr.device)
    obs, glb, _ = tr.vec.reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=tr.device)
    glb_t = torch.as_tensor(glb, dtype=torch.float32, device=tr.device)

    log_path = os.path.join(out, "train_log.csv")
    new_log = tr.updates == 0
    if new_log:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["update", "env_steps", "ret", "pg", "v", "ent", "kl", "clipfrac",
                 "gnorm", "sps",
                 "task_E", "task_P", "task_R", "task_C",
                 "goal_exp_arr", "goal_exp_done", "goal_pat_arr", "goal_pat_band",
                 "goal_relay_arr", "goal_chg_arr", "mean_d_init"])

    steps_per_update = args.rollout * tr.vec.n_envs
    n_updates = max(1, (args.total_steps - tr.env_steps) // steps_per_update)
    t_start, t_lastsave = time.time(), time.time()
    # 目标指标累计器（每个 update 重置，监控近期趋势）
    task_cnt = {"E": 0, "P": 0, "R": 0, "C": 0}
    goal_cnt = {"exp_arr": 0, "exp_done": 0, "exp_n": 0, "pat_arr": 0, "pat_band": 0,
                "pat_n": 0, "relay_arr": 0, "relay_n": 0, "chg_arr": 0, "chg_n": 0}

    print(f"[train] {args.trial_id} device={tr.device} envs={tr.vec.n_envs} "
          f"workers={getattr(tr.vec, 'n_workers', 1)} steps/update={steps_per_update} "
          f"resume_steps={tr.env_steps}")

    for upd in range(1, n_updates + 1):
        buf.start(obs_t, glb_t)
        ep_ret = 0.0
        t0 = time.time()
        for _ in range(args.rollout):
            act, logp, val = tr.act(obs_t, glb_t)
            obs, rew, done, glb, infos = tr.vec.step(act.cpu().numpy())
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=tr.device)
            glb_t = torch.as_tensor(glb, dtype=torch.float32, device=tr.device)
            rew_t = torch.as_tensor(rew, dtype=torch.float32, device=tr.device)
            done_t = torch.as_tensor(done.astype(np.float32), device=tr.device)
            buf.add(obs_t, glb_t, act, logp, val, rew_t, done_t)
            ep_ret += float(rew.sum())
            # ---- 训练内监控指标（来自实际轨迹）----
            for info in infos:
                tk = info["tasks"]
                mk = info.get("modes", [])
                for i, t_ in enumerate(tk):
                    if t_ == 0:
                        key = "P" if (mk[i] if i < len(mk) else 0) == 1 else "E"
                    elif t_ == 1:
                        key = "R"
                    else:
                        key = "C"
                    task_cnt[key] += 1
                ge = info.get("goal_events") or {}
                for i, ev in ge.items():
                    if "d_now" not in ev:
                        continue
                    t_ = tk[i] if i < len(tk) else -1
                    m_ = mk[i] if i < len(mk) else 0
                    if t_ == 0 and m_ == 1:
                        goal_cnt["pat_n"] += 1
                        goal_cnt["pat_arr"] += int(bool(ev.get("g_arrived")))
                        goal_cnt["pat_band"] += int(bool(ev.get("in_band")))
                    elif t_ == 0:
                        goal_cnt["exp_n"] += 1
                        goal_cnt["exp_arr"] += int(bool(ev.get("g_arrived")))
                        goal_cnt["exp_done"] += int(bool(ev.get("g_completed")))
                    elif t_ == 1:
                        goal_cnt["relay_n"] += 1
                        goal_cnt["relay_arr"] += int(bool(ev.get("g_arrived")))
                    elif t_ == 2:
                        goal_cnt["chg_n"] += 1
                        goal_cnt["chg_arr"] += int(bool(ev.get("g_arrived")))

        t_roll = time.time() - t0
        tu = time.time()
        with torch.no_grad():
            _, _, last_val = tr.act(obs_t, glb_t)
        buf.compute_gae(last_val, args.gamma, args.gae_lambda, tr.value_norm)
        # 熵系数线性退火（用保存的初值计算，不叠加）
        frac = min(1.0, tr.env_steps / max(1.0, args.entropy_anneal_steps))
        args.entropy_coef = tr.ent0 + frac * (args.entropy_final - tr.ent0)
        stats = tr.update(buf)
        t_upd = time.time() - tu

        tr.env_steps += steps_per_update
        tr.updates += 1
        sps = steps_per_update / max(1e-6, time.time() - t0)

        if tr.updates % args.log_every == 0:
            def _r(a, b):
                return a / max(1, b)
            tot_t = sum(task_cnt.values())
            row = [tr.updates, tr.env_steps, ep_ret, stats["pg"], stats["v"], stats["ent"],
                   stats["kl"], stats["clipfrac"], stats["gnorm"], sps,
                   _r(task_cnt["E"], tot_t), _r(task_cnt["P"], tot_t),
                   _r(task_cnt["R"], tot_t), _r(task_cnt["C"], tot_t),
                   _r(goal_cnt["exp_arr"], goal_cnt["exp_n"]),
                   _r(goal_cnt["exp_done"], goal_cnt["exp_n"]),
                   _r(goal_cnt["pat_arr"], goal_cnt["pat_n"]),
                   _r(goal_cnt["pat_band"], goal_cnt["pat_n"]),
                   _r(goal_cnt["relay_arr"], goal_cnt["relay_n"]),
                   _r(goal_cnt["chg_arr"], goal_cnt["chg_n"]), 0.0]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"[{args.trial_id} {tr.updates:6d}] steps={tr.env_steps:>10d} "
                  f"ret={ep_ret:8.1f} pg={stats['pg']:+.4f} v={stats['v']:.4f} "
                  f"ent={stats['ent']:.3f} kl={stats['kl']:+.4f} "
                  f"cf={stats['clipfrac']:.3f} gn={stats['gnorm']:.2f} | "
                  f"E/P/R/C={row[10]:.2f}/{row[11]:.2f}/{row[12]:.2f}/{row[13]:.2f} "
                  f"arr(x/p/r/c)={row[14]:.2f}/{row[16]:.2f}/{row[18]:.2f}/{row[19]:.2f} "
                  f"| {sps:.0f} sps | {(time.time()-t_start)/60:.0f}min", flush=True)
            task_cnt = {k: 0 for k in task_cnt}
            goal_cnt = {k: 0 for k in goal_cnt}

        if time.time() - t_lastsave > args.save_every_sec or tr.env_steps >= args.total_steps:
            tr.save_ckpt(os.path.join(out, "ckpt_latest.pt"))
            tr.save_actor(os.path.join(out, "actor_latest.pt"))
            t_lastsave = time.time()

    tr.save_ckpt(os.path.join(out, "ckpt_latest.pt"))
    tr.save_actor(os.path.join(out, "actor_latest.pt"))
    tr.vec.close()
    print(f"[done] {args.trial_id} env_steps={tr.env_steps}")


if __name__ == "__main__":
    main()
