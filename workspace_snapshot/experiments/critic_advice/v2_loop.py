import sys, os, json, multiprocessing as mp
sys.path.insert(0, "experiments/critic_advice")
import numpy as np, torch
torch.set_num_threads(1)
from harness import make_env, load_actor, make_h_brain, task_metrics, BENCH
from gen_data import regime_cfg
from policy_experiment import brain_variant
from critic_data import ctx_v2, build_features
from exec_model import StandardModel, make_mlp

CFG_NAMES = ["P1_relay2", "P2_explore85", "P3a_charge1", "P4a_chargeat200", "P4b_chargeat320"]
CONFIGS = {"P1_relay2": dict(n_relay_min=2), "P2_explore85": dict(explore_until=0.85),
           "P3a_charge1": dict(max_chargers=1), "P4a_chargeat200": dict(charge_at=200.0),
           "P4b_chargeat320": dict(charge_at=320.0)}
SWITCHES = [0, 100, 200, 300, 400]

ck = torch.load("experiments/critic_advice/runs/critic_models/switch_selector.pt",
                map_location="cpu", weights_only=False)
nets = []
for o in ck["nets"]:
    net = make_mlp(580, len(CFG_NAMES), 128)
    net.load_state_dict(o["net"]); net.eval()
    nets.append(StandardModel(net, dict(mu=o["mu"], sd=o["sd"], ymu=o["ymu"], ysd=o["ysd"])))

def run_v2(seed, regime, policy):
    cfg = regime_cfg(regime)
    env = make_env(seed, cfg); env.reset()
    actor = load_actor(BENCH["actor"]); brain = make_h_brain(cfg)
    replan = cfg["replan_every"]
    obs, cov, done = None, [], False
    cur_kw, rng = {}, np.random.RandomState(seed)
    n_switch = 0
    while env.t < 600 and not done:
        if env.t % replan == 0:
            if env.t in SWITCHES:
                s_ctx = None
                from lowlevel.mission_demo import sense_state
                s = sense_state(env)
                plan, _, _ = brain.plan(s)
                obs = env.set_plan(plan)
                ctx = ctx_v2(env)
                X = build_features(ctx, ctx["orders"], ctx["orders"])[None]
                if policy == "selector":
                    p = np.mean([m.predict(X) for m in nets], 0).ravel()
                    j = int(np.argmax(p)) if p.max() > 0 else -1
                    cur_kw = CONFIGS[CFG_NAMES[j]] if j >= 0 else {}
                    n_switch += int(j >= 0)
                elif policy == "random_sw":
                    if rng.random() < 0.5:
                        cur_kw = CONFIGS[CFG_NAMES[rng.randint(5)]]
                        n_switch += 1
                    else:
                        cur_kw = {}
                elif policy == "base":
                    cur_kw = {}
                plan = brain_variant(s, **cur_kw)
            else:
                from lowlevel.mission_demo import sense_state
                s = sense_state(env)
                plan = brain_variant(s, **cur_kw)
            obs = env.set_plan(plan)
        with torch.no_grad():
            a = actor.act(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)[0].numpy()
        obs, _r, done, info = env.step(a)
        cov.append(float(info.get("coverage_rate", 0.0)))
    m = task_metrics(env)
    return dict(seed=seed, regime=regime, policy=policy,
                ext=float(m["extinguished_cells"]), burned=float(m["burned_cells"]),
                contained=float(m["contained"]), rknown=float(m["rknown"]),
                cov_mean=float(np.mean(cov)) if cov else 0.0,
                explored=float(m["explored"]), depleted=float(m["depleted"]),
                n_switch=n_switch)

def worker(a):
    try:
        return run_v2(*a)
    except Exception as e:
        import traceback
        return dict(seed=a[0], regime=a[1], policy=a[2], error=str(e),
                    trace=traceback.format_exc()[-500:])

if __name__ == "__main__":
    jobs = []
    for regime, s0 in (("normal", 50000), ("scarce", 51000)):
        for p in ("base", "random_sw", "selector"):
            for i in range(60):
                jobs.append((s0 + i, regime, p))
    with mp.get_context("fork").Pool(24) as pool:
        res = pool.map(worker, jobs)
    ok = [r for r in res if "error" not in r]
    for r in res[:3]:
        if "error" in r:
            print("ERR", r["policy"], r["regime"], r["seed"], r["error"], r.get("trace","")[-300:])
    with open("experiments/critic_advice/runs/critic_v2_loop.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import defaultdict
    by = defaultdict(dict)
    for r in ok:
        by[(r["regime"], r["policy"])][r["seed"]] = r
    print("\n=== v2 闭环（60 种子配对，Δ = 臂 − base）===")
    for regime in ("normal", "scarce"):
        base = by[(regime, "base")]
        for p in ("random_sw", "selector"):
            arm = by[(regime, p)]
            common = [s for s in arm if s in base]
            for k in ("ext", "burned", "contained", "rknown", "cov_mean"):
                d = np.array([arm[s][k] - base[s][k] for s in common], float)
                ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
                print(f"  {regime:<7s} {p:<10s} Δ{k:<10s} {d.mean():+8.2f}±{ci:5.2f} (n={len(d)})")
            print(f"    平均切换次数 {np.mean([arm[s]['n_switch'] for s in common]):.1f}/5")
