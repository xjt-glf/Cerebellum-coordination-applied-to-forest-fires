import sys, os
sys.path.insert(0, "experiments/critic_advice")
import numpy as np, torch, json, multiprocessing as mp
torch.set_num_threads(1)
from harness import make_env, load_actor, task_metrics, BENCH
from gen_data import regime_cfg
from policy_experiment import brain_variant

CONFIGS = {
    "P0_base": dict(),
    "P1_relay2": dict(n_relay_min=2),
    "P2_explore85": dict(explore_until=0.85),
    "P3a_charge1": dict(max_chargers=1),
    "P4a_chargeat200": dict(charge_at=200.0),
    "P4b_chargeat320": dict(charge_at=320.0),
}

def run_with_switch(seed, regime, cfg_name, t_switch):
    cfg = regime_cfg(regime)
    env = make_env(seed, cfg); env.reset()
    actor = load_actor(BENCH["actor"])
    replan = cfg["replan_every"]; max_steps = cfg["steps"]
    kw = CONFIGS[cfg_name]
    obs, cov, done = None, [], False
    while env.t < max_steps and not done:
        if env.t % replan == 0:
            from lowlevel.mission_demo import sense_state
            s = sense_state(env)
            use = kw if (cfg_name != "P0_base" and env.t >= t_switch) else dict()
            plan = brain_variant(s, **use)
            obs = env.set_plan(plan)
        with torch.no_grad():
            a = actor.act(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)[0].numpy()
        obs, _r, done, info = env.step(a)
        cov.append(float(info.get("coverage_rate", 0.0)))
    m = task_metrics(env)
    return dict(ext=float(m["extinguished_cells"]), burned=float(m["burned_cells"]),
                contained=float(m["contained"]), rknown=float(m["rknown"]),
                cov_mean=float(np.mean(cov)) if cov else 0.0)

def worker(a):
    seed, regime, cfg, ts = a
    try:
        return (seed, regime, cfg, ts, run_with_switch(seed, regime, cfg, ts))
    except Exception as e:
        return (seed, regime, cfg, ts, dict(error=str(e)))

if __name__ == "__main__":
    SWITCHES = [0, 100, 200, 300, 400]
    jobs = []
    for regime, s0 in (("normal", 48000), ("scarce", 49000)):
        for i in range(40):
            for cfg in CONFIGS:
                for ts in SWITCHES:
                    jobs.append((s0 + i, regime, cfg, ts))
    ctx = mp.get_context("fork")
    with ctx.Pool(24) as pool:
        res = pool.map(worker, jobs)
    out = {}
    for seed, regime, cfg, ts, m in res:
        if "error" in m:
            print("ERR", seed, regime, cfg, ts, m["error"]); continue
        out.setdefault(regime, {}).setdefault(cfg, {}).setdefault(ts, {})[seed] = m
    # 基线 P0@t=0 就是 H
    for regime in ("normal", "scarce"):
        base = {s: out[regime]["P0_base"][0][s]["ext"] for s in out[regime]["P0_base"][0]}
        seeds = sorted(base)
        print(f"\n=== {regime}: oracle 切换收益（Δext vs H，n=40，切换点×配置）===")
        best_fixed = None
        for cfg in CONFIGS:
            if cfg == "P0_base": continue
            row = []
            for ts in SWITCHES:
                d = [out[regime][cfg][ts][s]["ext"] - base[s] for s in seeds]
                row.append(np.mean(d))
            print(f"  {cfg:<16s} " + " ".join(f"{v:+6.1f}" for v in row))
        # oracle：每个种子选最优 (cfg, ts)
        gains = []
        for s in seeds:
            cands = [out[regime][cfg][ts][s]["ext"] - base[s]
                     for cfg in CONFIGS if cfg != "P0_base" for ts in SWITCHES]
            gains.append(max(cands))
        gains = np.array(gains)
        sm = gains
        ci = 1.96 * sm.std(ddof=1)/np.sqrt(len(sm)) if len(sm)>1 else float('nan')
        print(f"  oracle(种子内最优 cfg×ts): {sm.mean():+.2f}±{ci:.2f}")
        # oracle 固定 (cfg,ts)（全体种子同一选择，更接近可部署）
        allg = {}
        for cfg in CONFIGS:
            if cfg == "P0_base": continue
            for ts in SWITCHES:
                allg[(cfg, ts)] = np.mean([out[regime][cfg][ts][s]["ext"] - base[s] for s in seeds])
        k = max(allg, key=allg.get)
        print(f"  最优固定 (cfg,ts) = {k}: {allg[k]:+.2f}")
    with open("experiments/critic_advice/runs/oracle_switch.json", "w") as f:
        json.dump({rg: {c: {str(t): {str(s): m for s, m in d.items()}
                           for t, d in v.items()} for c, v in out[rg].items()}
                   for rg in out}, f)
    print("saved oracle_switch.json")
