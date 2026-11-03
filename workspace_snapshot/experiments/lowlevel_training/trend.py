# -*- coding: utf-8 -*-
"""训练趋势速览：每个 trial 的最近 5 行目标指标（arr/done 为任务步中的已到访/已完成占比）。"""
import csv, os, sys

ROOT = "/ssd3/xiangjutao/hrl_dqn_improve/experiments/lowlevel_training/runs"
trials = sys.argv[1:] or ["P1_s1101", "P1_s1102", "P2_s1201", "P2_s1202",
                          "P3_s1301", "P3_s1302", "A_roles_s1401", "A_reward_s1501"]
for t in trials:
    p = os.path.join(ROOT, t, "train_log.csv")
    if not os.path.exists(p):
        print(f"{t}: no log"); continue
    rows = list(csv.DictReader(open(p)))
    if not rows:
        print(f"{t}: empty"); continue
    n = len(rows)
    take = rows[-3:]
    def m(k, r):
        try: return float(r[k])
        except Exception: return float("nan")
    print(f"== {t} (updates={n}, steps={rows[-1]['env_steps']})")
    for r in take:
        print(f"   steps={int(float(r['env_steps'])):>9} ret={float(r['ret']):8.1f} "
              f"ent={float(r['ent']):.3f} kl={float(r['kl']):+.4f} cf={float(r['clipfrac']):.3f} "
              f"E/P/R/C={float(r['task_E']):.2f}/{float(r['task_P']):.2f}/{float(r['task_R']):.2f}/{float(r['task_C']):.2f} "
              f"goal: exp_arr={float(r['goal_exp_arr']):.2f} exp_done={float(r['goal_exp_done']):.2f} "
              f"pat_arr={float(r['goal_pat_arr']):.2f} pat_band={float(r['goal_pat_band']):.2f} "
              f"rly_arr={float(r['goal_relay_arr']):.2f} chg_arr={float(r['goal_chg_arr']):.2f}")
