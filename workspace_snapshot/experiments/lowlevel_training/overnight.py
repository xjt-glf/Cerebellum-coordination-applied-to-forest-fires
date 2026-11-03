# -*- coding: utf-8 -*-
"""整夜训练调度器（H200 计划 §11 入口合同）。

    python experiments/lowlevel_training/overnight.py --plan research_outputs/h200_night_plan.json \
        --wall-hours 8 launch|status|monitor|stop|eval

launch : 按 plan 的 initial_trials 启动 8 个训练进程（一卡一 trial，CUDA_VISIBLE_DEVICES 映射）。
monitor: 守护循环：进程健康、步速、日志尾部，写入 runs_dir/monitor.log。
status : 打印全部 trial 状态。
stop   : 停止全部训练进程。
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))


def load_plan(path):
    with open(path) as f:
        return json.load(f)


def trial_cmd(plan, t, wall_hours):
    p = os.path.join(_ROOT, "experiments", "lowlevel_training", "train.py")
    init = {"actor_only": "actor_only", "random": "random"}.get(t["init"], "actor_only")
    bc = t["init"] in ("actor_then_bc", "random_then_bc")
    runs_root = os.path.join(_ROOT, plan["output"]["runs_dir"]) \
        if plan.get("output", {}).get("runs_dir") \
        else os.path.join(_ROOT, "experiments", "lowlevel_training", "runs")
    cmd = [sys.executable, p,
           "--trial-id", t["id"],
           "--run-root", runs_root,
           "--gpu", str(t["gpu"]),
           "--reward", t["reward"] if t["reward"] in ("P1", "P2") else "original",
           "--roles", t["roles"],
           "--init", "bc" if bc else init,
           "--seed", str(t["seed"]),
           "--actor-lr", str(t.get("actor_lr", 1e-4)),
           "--total-steps", "40000000"]
    if bc:
        cmd += ["--bc-data", os.path.join(_ROOT, "experiments", "lowlevel_training",
                                          "runs", "teacher_bc.npz")]
    return cmd, runs_root


def launch(plan, args):
    runs_dir = os.path.join(_ROOT, plan["output"]["runs_dir"])
    os.makedirs(runs_dir, exist_ok=True)
    pids = {}
    for t in plan["initial_trials"]:
        out = os.path.join(runs_dir, t["id"])
        os.makedirs(out, exist_ok=True)
        cmd = trial_cmd(plan, t, args.wall_hours)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(t["gpu"])
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        logf = open(os.path.join(out, "stdout.log"), "a")
        p = subprocess.Popen(cmd, cwd=_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
        pids[t["id"]] = p.pid
        print(f"[launch] {t['id']} gpu={t['gpu']} pid={p.pid} reward={t['reward']} "
              f"roles={t['roles']} init={t['init']} seed={t['seed']}")
        time.sleep(2)
    with open(os.path.join(runs_dir, "pids.json"), "w") as f:
        json.dump(pids, f, indent=2)
    with open(os.path.join(runs_dir, "launch_meta.json"), "w") as f:
        json.dump(dict(launched_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                        wall_hours=args.wall_hours,
                        trials=plan["initial_trials"]), f, indent=2)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_last_line(path):
    try:
        with open(path, "rb") as f:
            lines = f.read().decode("utf-8", "replace").strip().splitlines()
            return lines[-1] if lines else ""
    except OSError:
        return ""


def status(plan, verbose=True):
    runs_dir = os.path.join(_ROOT, plan["output"]["runs_dir"])
    try:
        pids = json.load(open(os.path.join(runs_dir, "pids.json")))
    except OSError:
        pids = {}
    rows = []
    for t in plan["initial_trials"]:
        tid = t["id"]
        out = os.path.join(runs_dir, tid)
        log = os.path.join(out, "stdout.log")
        alive = pid_alive(pids.get(tid, -1))
        line = read_last_line(log)
        rows.append((tid, alive, line))
        if verbose:
            print(f"{'RUN ' if alive else 'DEAD'} {tid}: {line[:150]}")
    return rows


def monitor(plan, args):
    runs_dir = os.path.join(_ROOT, plan["output"]["runs_dir"])
    mlog = os.path.join(runs_dir, "monitor.log")
    t_end = time.time() + args.wall_hours * 3600 - args.reserve_minutes * 60
    while time.time() < t_end:
        rows = status(plan, verbose=False)
        with open(mlog, "a") as f:
            f.write(f"==== {time.strftime('%H:%M:%S')} ====\n")
            for tid, alive, line in rows:
                f.write(f"{'RUN ' if alive else 'DEAD'} {tid}: {line[:200]}\n")
        n_dead = sum(0 if a else 1 for _, a, _ in rows)
        if n_dead == len(rows):
            print("[monitor] all trials stopped.")
            break
        time.sleep(args.interval)


def stop(plan):
    runs_dir = os.path.join(_ROOT, plan["output"]["runs_dir"])
    try:
        pids = json.load(open(os.path.join(runs_dir, "pids.json")))
    except OSError:
        pids = {}
    for tid, pid in pids.items():
        if pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"[stop] {tid} ({pid}) SIGTERM")
            except OSError:
                pass
    time.sleep(8)
    for tid, pid in pids.items():
        if pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"[stop] {tid} ({pid}) SIGKILL")
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=os.path.join(_ROOT, "research_outputs",
                                                   "h200_night_plan.json"))
    ap.add_argument("--wall-hours", type=float, default=8)
    ap.add_argument("--reserve-minutes", type=float, default=60)
    ap.add_argument("--interval", type=float, default=120)
    ap.add_argument("cmd", nargs="?", default="status",
                    choices=["launch", "status", "monitor", "stop"])
    args = ap.parse_args()
    plan = load_plan(args.plan)
    {"launch": lambda: launch(plan, args),
     "status": lambda: status(plan),
     "monitor": lambda: monitor(plan, args),
     "stop": lambda: stop(plan)}[args.cmd]()


if __name__ == "__main__":
    main()
