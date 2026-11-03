# -*- coding: utf-8 -*-
"""用训练好的低层策略跑一次完整任务仿真（可指定环境参数，可输出 GIF）。

    # 默认：100x100、3 个火点、10 名人员、4 架无人机、600 步
    python3 deliverable/simulate.py

    # 换参数 + 出 GIF
    python3 deliverable/simulate.py --fires 4 --rescuers 12 --uavs 6 \
            --steps 800 --seed 7 --gif out.gif

    # 多回合统计（不出图，看均值）
    python3 deliverable/simulate.py --episodes 5

上层（"大脑"）用的是 lowlevel/mission_demo.py 里手写的规则规划器 `brain()`，
每 replan-every 步以 **LLM Task Planner 的 JSON 格式** 下发全队指令；
接真正的大模型时只需把 brain() 换成模型调用，低层权重不用动。
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("环境参数")
    g.add_argument("--fires", type=int, default=3, help="初始火点数量")
    g.add_argument("--rescuers", type=int, default=10, help="地面救援人员数量")
    g.add_argument("--uavs", type=int, default=4, help="无人机数量（1-8）")
    g.add_argument("--grid", type=int, default=100, help="任务区域边长（格）")
    g.add_argument("--steps", type=int, default=600, help="一回合的时间步数")
    g.add_argument("--spread-steps", type=int, default=40, help="开局前火自行蔓延的步数")
    g.add_argument("--spread-gain", type=float, default=1.0, help="火蔓延速度增益")
    g.add_argument("--seed", type=int, default=20000, help="随机种子")
    g.add_argument("--random-fires", action="store_true",
                   help="火点数每回合在 2-4 之间随机（默认按 --fires 固定）")

    g2 = p.add_argument_group("上层规划")
    g2.add_argument("--replan-every", type=int, default=20, help="上层多少步重规划一次")
    g2.add_argument("--charge-at", type=float, default=260.0,
                    help="电量低于该值就排队去补给（满电 600）")
    g2.add_argument("--explore-until", type=float, default=0.70,
                    help="全图探索率达到该值后，感知任务从搜寻转为绕火点巡逻")

    g3 = p.add_argument_group("运行方式")
    g3.add_argument("--actor", default=os.path.join(_HERE, "model", "actor_best.pt"))
    g3.add_argument("--episodes", type=int, default=1, help=">1 时只统计均值，不出图")
    g3.add_argument("--gif", default=None, help="输出 GIF 路径（留空则不出图）")
    g3.add_argument("--fps", type=int, default=6, help="GIF 帧率")
    g3.add_argument("--every", type=int, default=3, help="每几步录一帧")
    g3.add_argument("--random-policy", action="store_true",
                    help="用随机动作代替策略，作为对照基线")
    g3.add_argument("--quiet", action="store_true", help="不打印每次重规划的细节")
    return p.parse_args()


def run_episode(args, seed, record=False):
    """跑一回合。record=True 时同时抓帧。返回 (指标 dict, 帧列表, 指令日志)。"""
    import matplotlib
    if record:
        matplotlib.use("Agg")
    import torch
    from lowlevel.env_wrapper import TaskConditionedEnv
    from lowlevel.models import Actor
    from lowlevel.obs_builder import OBS_DIM
    from lowlevel.mission_demo import sense_state, brain, grab

    env = TaskConditionedEnv(
        num_uavs=args.uavs, num_rescuers=args.rescuers,
        n_fire_seeds=args.fires,
        n_fire_seeds_range=((2, 4) if args.random_fires else None),
        grid_size=(args.grid, args.grid), max_steps=args.steps,
        spread_steps=args.spread_steps, spread_gain=args.spread_gain,
        seed=seed, replan_every=0)          # 重规划由本脚本接管
    env.reset()

    actor = None
    if not args.random_policy:
        sd = torch.load(args.actor, map_location="cpu")
        actor = Actor(OBS_DIM, 9, hidden=int(sd["head.weight"].shape[1]))
        actor.load_state_dict(sd)
        actor.eval()

    rng = np.random.RandomState(seed)
    frames, log, cov = [], [], []
    obs, tot = None, 0.0
    size = (800, 800)
    for t in range(args.steps):
        if t % args.replan_every == 0:
            s = sense_state(env)
            plan, reason = brain(s, explore_until=args.explore_until,
                                 charge_at=args.charge_at)
            obs = env.set_plan(plan)
            log.append((t, s, plan, reason))
            if not args.quiet:
                print(f"[t={t:4d}] 探索率={s['explored']:.3f} 发现率={s['discovery']:.3f} "
                      f"覆盖率={s['coverage']:.3f} 人员已知={s['rescuer_known']:.3f} "
                      f"未合围火点={len(s['clusters'])} 过火={s['burned']:.4f}")
                for r in reason:
                    print("   ", r)
        if actor is None:
            a = rng.randint(0, 9, size=args.uavs)
        else:
            with torch.no_grad():
                a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                              deterministic=True)[0].numpy()
        obs, r, done, info = env.step(a)
        tot += float(np.sum(r))
        cov.append(info["coverage_rate"])
        if record and (t % args.every == 0 or done):
            env.env.render(pause_time=0.001)
            frames.append(grab(env.env.fig, size))
        if done:
            break

    metrics = dict(
        steps=t + 1, ret=tot,
        explored=info["explored_rate"],
        discovery=info["fire_discovery_rate"],
        rescuer_known=info["rescuer_known_fire_rate"],
        coverage=float(np.mean(cov)),
        burned=info["burned_area_rate"],
        extinguished=info["n_extinguished"],
        depleted=info["n_depleted"],
        contained=info["n_contained"],
        active_fires=info["n_clusters"],
    )
    return metrics, frames, log


HEAD = ("步数", "回报", "全图探索率", "火情发现率", "人员已知火情",
        "无人机覆盖率", "过火面积", "扑灭格数", "已合围火点", "剩余活火点", "耗尽机数")
KEYS = ("steps", "ret", "explored", "discovery", "rescuer_known",
        "coverage", "burned", "extinguished", "contained", "active_fires", "depleted")


def show(tag, ms):
    """ms: list of metrics dict"""
    a = {k: np.mean([m[k] for m in ms]) for k in KEYS}
    print(f"\n=== {tag}（{len(ms)} 回合均值）===")
    for h, k in zip(HEAD, KEYS):
        v = a[k]
        fmt = "%.4f" if k == "burned" else ("%.3f" if v < 10 else "%.1f")
        print(f"  {h:12s} {fmt % v}")


def main():
    args = parse_args()
    if not (1 <= args.uavs <= 8):
        raise SystemExit("--uavs 必须在 1-8 之间（观测里的 agent-id one-hot 上限是 8）")
    if not os.path.exists(args.actor) and not args.random_policy:
        raise SystemExit(f"找不到权重 {args.actor}")

    print(f"环境：{args.grid}x{args.grid} | 火点 {args.fires}"
          f"{'(每回合 2-4 随机)' if args.random_fires else ''} | "
          f"人员 {args.rescuers} | 无人机 {args.uavs} | 回合 {args.steps} 步 | "
          f"策略 {'随机基线' if args.random_policy else os.path.basename(args.actor)}")

    record = bool(args.gif) and args.episodes == 1
    ms = []
    for k in range(args.episodes):
        m, frames, log = run_episode(args, args.seed + k, record=record)
        ms.append(m)
        if args.episodes > 1:
            print(f"  回合 {k + 1}/{args.episodes}: 探索 {m['explored']:.3f} "
                  f"发现 {m['discovery']:.3f} 扑灭 {m['extinguished']} "
                  f"耗尽 {m['depleted']}")
    show("仿真结果", ms)

    if record and frames:
        from PIL import Image
        os.makedirs(os.path.dirname(os.path.abspath(args.gif)) or ".", exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(1, args.fps)), loop=0, optimize=True)
        print(f"\n[GIF] {args.gif}（{len(frames)} 帧 / {args.fps} fps）")
        md = os.path.splitext(args.gif)[0] + "_commands.md"
        import json
        with open(md, "w", encoding="utf-8") as f:
            f.write("# 上层指令日志（每 %d 步重规划一次）\n\n" % args.replan_every)
            for t0, s, plan, reason in log:
                f.write(f"## t={t0}\n\n- 态势：探索率 {s['explored']:.3f}、"
                        f"火情发现率 {s['discovery']:.3f}、覆盖率 {s['coverage']:.3f}、"
                        f"人员已知火情 {s['rescuer_known']:.3f}、"
                        f"未合围火点 {len(s['clusters'])} 处\n")
                for r in reason:
                    f.write(f"- 决策：{r}\n")
                f.write("\n```json\n" + json.dumps(plan, ensure_ascii=False, indent=2)
                        + "\n```\n\n")
        print(f"[指令日志] {md}")


if __name__ == "__main__":
    main()
