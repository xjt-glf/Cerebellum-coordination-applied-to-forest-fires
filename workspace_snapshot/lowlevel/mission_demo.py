# -*- coding: utf-8 -*-
"""完整任务演示：上层（"大脑"）集中式下发指令，低层强化学习策略执行，录成 GIF。

    python3 lowlevel/mission_demo.py --actor lowlevel/runs/w1_ent02/actor_best.pt \
            --out lowlevel/runs/pic/mission.gif

上层的每一次决策都以 **LLM Task Planner 的 JSON 格式** 下发，走的是
`TaskConditionedEnv.set_plan()` → `TeamPlan.from_llm_json()` 这条唯一接缝，
所以把这里的 `brain()` 换成真正的大模型调用即可，低层不用改。

环境自带的 render() 是交互式的（plt.pause），这里强制 Agg 后端直接从 canvas
取帧，不需要显示器。
"""
import argparse
import json
import math
import os
import sys

# 可视化后端可选：容器训练/评估环境无 matplotlib 时也能 import 本模块
# （brain/sense_state 不依赖它；仅 main() 录 GIF 需要）。
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:  # pragma: no cover
    matplotlib = None
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowlevel.env_wrapper import TaskConditionedEnv
from lowlevel.models import Actor
from lowlevel.obs_builder import OBS_DIM, shared_cache, assigned_station


def sense_state(env):
    """把环境状态整理成"大脑"可读的简报（真正接 LLM 时这就是 prompt 的内容）。"""
    e = env.env
    c = shared_cache(e)
    H, W = e.grid_size
    return dict(
        t=env.t, H=H, W=W,
        charging_stations=[list(map(float, st)) for st in e.charging_stations],
        base_pos=list(map(float, e.base_pos)),
        explored=round(c["explored_rate"], 3),
        discovery=round(float(getattr(e, "fire_discovery_rate", 0.0)), 3),
        burned=round(float(getattr(e, "burned_area_rate", 0.0)), 4),
        coverage=round(float(getattr(e, "coverage_rate", 0.0)), 3),
        rescuer_known=round(float(getattr(e, "rescuer_known_fire_rate", 0.0)), 3),
        max_aoi=int(getattr(e, "max_AoI", 0)),
        clusters=[dict(centroid=[round(v, 1) for v in cl["centroid"]], size=cl["size"])
                  for cl in c["clusters"]],
        rescuers=[dict(id=int(r.rescuer_id), pos=[round(r.x, 1), round(r.y, 1)],
                       covered=bool(r.is_covered), aoi=int(r.AoI)) for r in e.rescuers],
        uavs=[dict(id=i, pos=[round(u.x, 1), round(u.y, 1)], e=round(float(u.e)),
                   margin=round(float(u.e) - 2.5 * assigned_station(e, i)[1]))
              for i, u in enumerate(e.uavs)],
    )


# 搜寻阶段把任务区域切成 4 个象限，一机一块，避免全队挤在一起
QUADRANTS = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)]


def brain(s, explore_until=0.70, charge_at=300.0, max_chargers=2):
    """我作为"大脑"的集中式决策。返回 LLM 风格的 JSON（dict）。

    规则（按优先级）：
      1. 返航余量已经为负的必须去补给；其余按电量从低到高，在名额内（默认 2 台）
         把电量低于 charge_at 的派去补给。用**绝对电量**而不是返航余量做门限：
         余量是相对"当前分配到的站点"算的，而站点分配会随队友移动翻转，
         按余量下令会出现"忽然被判定还很充裕"然后错过窗口的情况（实测有 3 台耗尽在半路）。
         同时限名额，否则一次派 3 台去充电，覆盖率会直接崩。
      2. 中继：探索阶段留 1 台，铺开之后留 2 台，分别盯 AoI 最大且**彼此拉开**的人员。
         人员的火场认知完全来自无人机，中继断了他们就停在原地不灭火。
      3. 其余：全地图探索率没到 explore_until 就继续搜寻（按机号分到 4 个象限）；
         已经铺开就转巡逻，按机号轮转绑到不同火点，环半径交给 auto（按火点规模算）。
         探索率低于 0.95 时至少保留 1 台继续搜寻。
    """
    n = len(s["uavs"])
    orders = [None] * n
    reason = []

    # 补给排队：按余量从低到高，余量为负的必须走，其余在名额内做预防性补给。
    # 阈值必须给够：地图中心到最近站点约 55 格 = 137 点电，阈值给 40 等于
    # "等到飞不回去了才想起来充电"，实测会有 2 台直接耗尽在半路。
    # 同时限名额（默认 2 台），否则一次派 3 台去充电，覆盖率会直接崩。
    charging = []
    for u in sorted(s["uavs"], key=lambda z: z["e"]):
        if u["margin"] < 0:
            orders[u["id"]] = {"task": "charge"}
            charging.append((u["id"], f"返航余量 {u['margin']} 已为负"))
        elif len(charging) < max_chargers and u["e"] < charge_at:
            orders[u["id"]] = {"task": "charge"}
            charging.append((u["id"], f"电量 {u['e']} < {charge_at:.0f}"))
    for i, why in charging:
        reason.append(f"UAV{i} → 补给（{why}）")

    rest = [u["id"] for u in s["uavs"] if orders[u["id"]] is None]
    n_relay = 2 if (s["explored"] >= explore_until and len(rest) >= 3) else 1
    picked = []
    pool = [r for r in s["rescuers"] if not r["covered"]] or list(s["rescuers"])
    for _ in range(min(n_relay, len(rest))):
        cand = [r for r in pool
                if all(abs(r["pos"][0] - q["pos"][0]) + abs(r["pos"][1] - q["pos"][1]) > 20
                       for q in picked)]
        if not cand:
            break
        tr = max(cand, key=lambda r: r["aoi"])
        relay = min(rest, key=lambda i: abs(s["uavs"][i]["pos"][0] - tr["pos"][0])
                    + abs(s["uavs"][i]["pos"][1] - tr["pos"][1]))
        orders[relay] = {"task": "relay", "rescuer_id": tr["id"], "handover": True}
        rest.remove(relay)
        picked.append(tr)
        reason.append(f"UAV{relay} → 中继人员#{tr['id']}"
                      f"(AoI={tr['aoi']}, covered={tr['covered']})")

    keep_explorer = s["explored"] < 0.95
    for k, i in enumerate(rest):
        want_explore = (s["explored"] < explore_until or not s["clusters"]
                        or (keep_explorer and k == 0))
        if want_explore:
            qx, qy = QUADRANTS[i % 4]
            orders[i] = {"task": "explore",
                         "target": [round(qx * s["H"], 1), round(qy * s["W"], 1)]}
            reason.append(f"UAV{i} → 搜寻象限 {orders[i]['target']}"
                          f"(探索率 {s['explored']:.2f})")
        else:
            cl = s["clusters"][k % len(s["clusters"])]
            orders[i] = {"task": "patrol", "target": cl["centroid"],
                         "ring_radius": "auto"}
            reason.append(f"UAV{i} → 巡逻火点 {cl['centroid']}(规模 {cl['size']})")

    return {"uavs": [o if o is not None else {"task": "explore"} for o in orders]}, reason


def grab(fig, size=None):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    if size is not None:
        from PIL import Image
        return np.asarray(Image.fromarray(buf).resize(size))
    return buf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", default="lowlevel/runs/w1_ent02/actor_best.pt")
    p.add_argument("--out", default="lowlevel/runs/pic/mission.gif")
    p.add_argument("--log", default=None, help="指令日志输出路径（默认与 gif 同名 .md）")
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--num-uavs", type=int, default=4)
    p.add_argument("--num-rescuers", type=int, default=8)
    p.add_argument("--fire-seeds", type=int, default=3)
    p.add_argument("--fixed-fires", action="store_true",
                   help="固定火点数（不按 2-4 随机），用于演示")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--replan-every", type=int, default=20)
    p.add_argument("--explore-until", type=float, default=0.70)
    p.add_argument("--every", type=int, default=5, help="每几步录一帧")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--frame-px", type=int, default=800)
    p.add_argument("--random-policy", action="store_true", help="录随机策略作为对照")
    p.add_argument("--charge-at", type=float, default=300.0)
    p.add_argument("--max-chargers", type=int, default=2)
    p.add_argument("--no-render", action="store_true", help="只跑不录，用于调上层策略")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    env = TaskConditionedEnv(num_uavs=args.num_uavs, max_steps=args.max_steps,
                             num_rescuers=args.num_rescuers,
                             n_fire_seeds=args.fire_seeds,
                             n_fire_seeds_range=(None if args.fixed_fires else (2, 4)),
                             seed=args.seed, replan_every=0)   # 重规划由本脚本接管
    env.reset()

    actor = None
    if not args.random_policy:
        sd = torch.load(args.actor, map_location="cpu")
        actor = Actor(OBS_DIM, 9, hidden=int(sd["head.weight"].shape[1]))
        actor.load_state_dict(sd)
        actor.eval()
        print(f"[loaded] {args.actor}")

    rng = np.random.RandomState(args.seed)
    frames, log, cov_hist = [], [], []
    obs = None
    size = (args.frame_px, args.frame_px)
    tot = 0.0
    for t in range(args.max_steps):
        if t % args.replan_every == 0:
            s = sense_state(env)
            plan, reason = brain(s, explore_until=args.explore_until,
                                 charge_at=args.charge_at,
                                 max_chargers=args.max_chargers)
            obs = env.set_plan(plan)          # 走 LLM JSON 这条接缝
            log.append((t, s, plan, reason))
            if not args.quiet:
                print(f"\n[t={t:3d}] 探索率={s['explored']:.3f} 发现率={s['discovery']:.3f} "
                  f"覆盖率={s['coverage']:.3f} 人员已知={s['rescuer_known']:.3f} "
                      f"火点={len(s['clusters'])} 过火={s['burned']:.4f}")
                for r in reason:
                    print("   ", r)
        if actor is None:
            a = rng.randint(0, 9, size=args.num_uavs)
        else:
            with torch.no_grad():
                a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                              deterministic=True)[0].numpy()
        obs, r, done, info = env.step(a)
        tot += float(np.sum(r))
        cov_hist.append(info["coverage_rate"])
        if (not args.no_render) and (t % args.every == 0 or done):
            env.env.render(pause_time=0.001)
            frames.append(grab(env.env.fig, size))
        if done:
            break

    print(f"\n[mission] steps={t + 1} return={tot:.1f} explored={info['explored_rate']:.3f} "
          f"discovery={info['fire_discovery_rate']:.3f} "
          f"rescuer_known={info['rescuer_known_fire_rate']:.3f} "
          f"coverage_mean={np.mean(cov_hist):.3f} burned={info['burned_area_rate']:.4f} "
          f"depleted={info['n_depleted']} "
          f"extinguished={int((env.env.grid['burning_state'] == 3).sum())}")

    if args.no_render:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(1, args.fps)), loop=0, optimize=True)
    print(f"[out] {args.out}  ({len(frames)} frames)")

    log_path = args.log or os.path.splitext(args.out)[0] + "_commands.md"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# 集中式指令日志（上层每 %d 步重规划一次）\n\n" % args.replan_every)
        for t0, s, plan, reason in log:
            f.write(f"## t={t0}\n\n")
            f.write(f"- 态势：探索率 {s['explored']:.3f}、火情发现率 {s['discovery']:.3f}、"
                    f"覆盖率 {s['coverage']:.3f}、人员已知火情 {s['rescuer_known']:.3f}、"
                    f"过火面积 {s['burned']:.4f}、已知火点 {len(s['clusters'])} 处\n")
            for r in reason:
                f.write(f"- 决策：{r}\n")
            f.write("\n```json\n" + json.dumps(plan, ensure_ascii=False, indent=2) + "\n```\n\n")
    print(f"[out] {log_path}")


if __name__ == "__main__":
    main()
