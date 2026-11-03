# -*- coding: utf-8 -*-
"""低层任务条件策略训练入口。

    python3 lowlevel/train_lowlevel.py --n-envs 8 --total-steps 2000000

产物：
    lowlevel/runs/<run_name>/ckpt_latest.pt   最新权重（含 optimizer，可续训）
    lowlevel/runs/<run_name>/ckpt_best.pt     评估回报最好的权重
    lowlevel/runs/<run_name>/actor_final.pt   仅 Actor（部署/接大模型时只需要这个）
    lowlevel/runs/<run_name>/train_log.csv    训练曲线
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowlevel.env_wrapper import TaskConditionedEnv
from lowlevel.vec_env import SubprocVecEnv, DummyVecEnv
from lowlevel.mappo import MAPPO, Config, RolloutBuffer, resolve_device
from lowlevel.reward import RewardConfig


def _reward_cfg(args):
    """势函数 shaping 的 gamma 必须与 GAE 的 gamma 一致，否则 shaping 不再是
    policy-invariant 的（Ng et al. 1999 的结论对 gamma 敏感）。"""
    cfg = RewardConfig()
    cfg.gamma = args.gamma
    return cfg


def make_env_fn(seed, args):
    def _f():
        return TaskConditionedEnv(
            num_uavs=args.num_uavs,
            max_steps=args.max_steps,
            spread_steps=args.spread_steps,
            grid_size=(args.grid, args.grid),
            n_fire_seeds=args.fire_seeds,
            spread_gain=args.spread_gain,
            num_rescuers=args.num_rescuers,
            seed=seed,
            task_sampler=args.task_sampler,
            replan_every=args.replan_every,
            reward_cfg=_reward_cfg(args),
        )
    return _f


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=None)
    p.add_argument("--num-uavs", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--spread-steps", type=int, default=40)
    p.add_argument("--grid", type=int, default=100)
    p.add_argument("--fire-seeds", type=int, default=3)
    p.add_argument("--spread-gain", type=float, default=1.0)
    p.add_argument("--num-rescuers", type=int, default=8)
    p.add_argument("--replan-every", type=int, default=25)
    p.add_argument("--task-sampler", default="planner",
                   choices=["planner", "mix", "mix+single"])
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--envs-per-worker", type=int, default=1,
                   help="每个采样进程承载的环境数（>1 可摊平 env.step 的重尾耗时）")
    p.add_argument("--rollout", type=int, default=200)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--n-minibatch", type=int, default=1)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--dummy-vec", action="store_true", help="单进程调试")
    p.add_argument("--resume", default=None)
    p.add_argument("--eval-every", type=int, default=25, help="每多少次更新做一次评估")
    p.add_argument("--eval-episodes", type=int, default=4)
    return p.parse_args()


EVAL_KEYS = ("ret", "explored", "discovery", "burned", "cov", "rknown", "ext", "depl")


def evaluate(policy, args, n_episodes=4, seed_base=10_000):
    """确定性评估：全地图探索率 / 火情发现率 / 过火面积 / 覆盖率 / 人员已知火情 /
    已扑灭格数 / 耗尽数。

    discovery / rknown / cov 取**回合均值**：人员的灭火能力增强后，回合末往往只剩
    个别燃烧格，而这两个比率的分母就是"当前真实燃烧格数"，末值会在极小分母上剧烈抖动
    （实测出现过 disc=0.000 但 explored=0.628 的情况）。explored / burned / ext
    是单调量，取末值。
    """
    device = policy.device
    res = []
    for k in range(n_episodes):
        env = TaskConditionedEnv(
            num_uavs=args.num_uavs, max_steps=args.max_steps,
            spread_steps=args.spread_steps, grid_size=(args.grid, args.grid),
            n_fire_seeds=args.fire_seeds, spread_gain=args.spread_gain,
            num_rescuers=args.num_rescuers, replan_every=args.replan_every,
            seed=seed_base + k, task_sampler=args.task_sampler,
            reward_cfg=_reward_cfg(args))
        obs = env.reset()
        tot = 0.0
        cov, disc, rk = [], [], []
        for _ in range(args.max_steps):
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            g = torch.as_tensor(env.global_features(), dtype=torch.float32,
                                device=device).unsqueeze(0)
            a, _, _ = policy.act(o, g, deterministic=True)
            obs, r, done, info = env.step(a[0].cpu().numpy())
            tot += float(np.sum(r))
            cov.append(info["coverage_rate"])
            disc.append(info["fire_discovery_rate"])
            rk.append(info["rescuer_known_fire_rate"])
            if done:
                break
        res.append((tot, info["explored_rate"], float(np.mean(disc)),
                    info["burned_area_rate"], float(np.mean(cov)),
                    float(np.mean(rk)), info["n_extinguished"], info["n_depleted"]))
        env.close()
    arr = np.array(res, dtype=np.float64)
    return {k: arr[:, j].mean() for j, k in enumerate(EVAL_KEYS)}


def main():
    args = parse_args()
    run = args.run_name or time.strftime("lowlevel_%m%d_%H%M")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", run)
    os.makedirs(out, exist_ok=True)

    cfg = Config()
    cfg.rollout, cfg.n_envs = args.rollout, args.n_envs
    cfg.total_env_steps = args.total_steps
    cfg.lr = cfg.critic_lr = args.lr
    cfg.gamma, cfg.gae_lambda, cfg.clip = args.gamma, args.gae_lambda, args.clip
    cfg.ppo_epochs, cfg.n_minibatch = args.ppo_epochs, args.n_minibatch
    cfg.entropy_coef, cfg.hidden, cfg.seed = args.entropy_coef, args.hidden, args.seed
    cfg.device = args.device

    np.random.seed(args.seed)
    env_fns = [make_env_fn(args.seed * 1000 + i, args) for i in range(args.n_envs)]
    vec = (DummyVecEnv(env_fns) if args.dummy_vec
           else SubprocVecEnv(env_fns, envs_per_worker=args.envs_per_worker))
    device = resolve_device(args.device)
    print(f"[init] device={device} obs_dim={vec.obs_dim} n_actions={vec.n_actions} "
          f"N={vec.num_uavs} global={vec.global_dim} envs={vec.n_envs} "
          f"workers={getattr(vec, 'n_workers', vec.n_envs)}")

    policy = MAPPO(vec.obs_dim, vec.n_actions, vec.global_dim, cfg)
    if args.resume and os.path.exists(args.resume):
        policy.load_state_dict(torch.load(args.resume, map_location=device))
        # optimizer state 里带着旧的 lr，续训时必须用命令行给的新 lr 覆盖回来
        for opt, lr in ((policy.opt_a, cfg.lr), (policy.opt_c, cfg.critic_lr)):
            for g in opt.param_groups:
                g["lr"] = lr
        print(f"[resume] {args.resume} (lr reset to {cfg.lr})")

    buf = RolloutBuffer(cfg.rollout, vec.n_envs, vec.num_uavs,
                        vec.obs_dim, vec.global_dim, device)

    obs, glb, _ = vec.reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    glb_t = torch.as_tensor(glb, dtype=torch.float32, device=device)

    log_path = os.path.join(out, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["update", "env_steps", "train_return", "pg", "v", "ent", "kl", "clipfrac",
             "eval_return", "eval_explored", "eval_discovery", "eval_burned",
             "eval_cov", "eval_rknown", "eval_ext", "eval_depl", "sps"])

    steps_per_update = cfg.rollout * vec.n_envs
    n_updates = max(1, cfg.total_env_steps // steps_per_update)
    best = -1e18          # 综合分（见下方 score 的定义）
    t_start = time.time()

    for upd in range(1, n_updates + 1):
        buf.start(obs_t, glb_t)
        ep_ret = 0.0
        t0 = time.time()
        t_act = t_env = 0.0
        for _ in range(cfg.rollout):
            ta = time.time()
            act, logp, val = policy.act(obs_t, glb_t)
            act_np = act.cpu().numpy()
            t_act += time.time() - ta
            ta = time.time()
            obs, rew, done, glb, infos = vec.step(act_np)
            t_env += time.time() - ta
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            glb_t = torch.as_tensor(glb, dtype=torch.float32, device=device)
            rew_t = torch.as_tensor(rew, dtype=torch.float32, device=device)
            done_t = torch.as_tensor(done.astype(np.float32), device=device)
            buf.add(obs_t, glb_t, act, logp, val, rew_t, done_t)
            ep_ret += float(rew.sum())

        t_roll = time.time() - t0
        tu = time.time()
        with torch.no_grad():
            _, _, last_val = policy.act(obs_t, glb_t)
        buf.compute_gae(last_val, cfg.gamma, cfg.gae_lambda, policy.value_norm)
        stats = policy.update(buf)
        t_upd = time.time() - tu

        env_steps = upd * steps_per_update
        sps = steps_per_update / max(1e-6, time.time() - t0)
        ev = {k: float("nan") for k in EVAL_KEYS}
        if upd % args.eval_every == 0 or upd == n_updates:
            ev = evaluate(policy, args, args.eval_episodes)
            # 最优权重按**综合分**挑，不按 eval_return 挑：回报会被覆盖项主导，
            # 出现"覆盖很好但不去探索"的版本也能拿最高回报（实测过）。
            score = (0.35 * ev["explored"] + 0.25 * ev["discovery"]
                     + 0.20 * ev["rknown"] + 0.15 * ev["cov"]
                     + 0.05 * min(1.0, ev["ext"] / 400.0)
                     - 0.10 * ev["depl"] / max(1, args.num_uavs))
            if score > best:
                best = score
                torch.save(policy.state_dict(), os.path.join(out, "ckpt_best.pt"))
                torch.save(policy.actor.state_dict(), os.path.join(out, "actor_best.pt"))

        if upd % cfg.log_interval == 0 or upd % args.eval_every == 0:
            print(f"[{upd:5d}/{n_updates}] steps={env_steps:>9d} "
                  f"ret/upd={ep_ret:9.1f} pg={stats['pg']:+.4f} v={stats['v']:.4f} "
                  f"ent={stats['ent']:.3f} kl={stats['kl']:+.4f} "
                  f"| eval ret={ev['ret']:.1f} exp={ev['explored']:.3f} "
                  f"disc={ev['discovery']:.3f} burn={ev['burned']:.4f} "
                  f"cov={ev['cov']:.3f} rk={ev['rknown']:.3f} ext={ev['ext']:.0f} "
                  f"depl={ev['depl']:.1f} "
                  f"| {sps:.0f} sps "
                  f"| act={t_act:.1f}s env={t_env:.1f}s upd={t_upd:.1f}s "
                  f"| {(time.time()-t_start)/60:.1f} min")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [upd, env_steps, ep_ret, stats["pg"], stats["v"], stats["ent"],
                 stats["kl"], stats["clipfrac"], ev["ret"], ev["explored"],
                 ev["discovery"], ev["burned"], ev["cov"], ev["rknown"],
                 ev["ext"], ev["depl"], sps])

        if upd % cfg.save_interval == 0 or upd == n_updates:
            torch.save(policy.state_dict(), os.path.join(out, "ckpt_latest.pt"))

    torch.save(policy.actor.state_dict(), os.path.join(out, "actor_final.pt"))
    vec.close()
    print(f"[done] {out}  best composite score = {best:.4f}")


if __name__ == "__main__":
    main()
