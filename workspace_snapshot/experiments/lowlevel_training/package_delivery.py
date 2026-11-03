# -*- coding: utf-8 -*-
"""交付打包：选中 Actor + 完整 checkpoint + 元信息 + 全部分支证据。"""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys

ROOT = "/ssd3/xiangjutao/hrl_dqn_improve"
RUNS = os.path.join(ROOT, "experiments/lowlevel_training/runs")
TRIALS_ALL = ["P1_s1101", "P1_s1102", "P2_s1201", "P2_s1202", "P3_s1301", "P3_s1302",
               "A_roles_s1401", "A_reward_s1501", "REL_s1303", "REL_s1304",
               "P2_s1203", "P2BC_s1204", "P2BC_s1205"]


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True, help="选中的 trial id")
    ap.add_argument("--actor", required=True, help="选中的 actor 文件")
    ap.add_argument("--ckpt", required=True, help="可继续训练的完整 checkpoint")
    ap.add_argument("--out", default=os.path.join(ROOT, "research_outputs",
                                                  "lowlevel_training_final"))
    ap.add_argument("--meta", default=None, help="补充元信息 JSON")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "branches"), exist_ok=True)
    os.makedirs(os.path.join(out, "evidence"), exist_ok=True)

    # 1) 选中 Actor + 完整 checkpoint
    shutil.copy2(args.actor, os.path.join(out, "actor_selected.pt"))
    shutil.copy2(args.ckpt, os.path.join(out, "ckpt_resumable.pt"))
    sha = sha256_of(os.path.join(out, "actor_selected.pt"))

    # 2) 每个分支的 config / 训练曲线 / 最终快照
    trial_table = []
    for t in TRIALS_ALL:
        d = os.path.join(RUNS, t)
        if not os.path.isdir(d):
            continue
        bd = os.path.join(out, "branches", t)
        os.makedirs(bd, exist_ok=True)
        for f in ("config.json", "train_log.csv", "stdout.log", "bc_info.json",
                  "actor_snapshot_gate1.pt", "actor_final_snapshot.pt",
                  "actor_pre_resume.pt", "actor_snap1.pt", "actor_snap2.pt"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(bd, f))
        log = os.path.join(d, "train_log.csv")
        steps = 0
        if os.path.exists(log):
            rows = list(csv.DictReader(open(log)))
            steps = int(float(rows[-1]["env_steps"])) if rows else 0
        cfg = {}
        cfgp = os.path.join(d, "config.json")
        if os.path.exists(cfgp):
            cfg = json.load(open(cfgp))
        trial_table.append(dict(
            id=t, env_steps=steps, reward=cfg.get("reward"), roles=cfg.get("roles"),
            init=cfg.get("init"), seed=cfg.get("seed"),
            actor_lr=cfg.get("actor_lr"),
            status=("selected" if t == args.trial else "stopped_at_gate")))
    with open(os.path.join(out, "branches", "trial_table.json"), "w") as f:
        json.dump(trial_table, f, indent=1)

    # 3) 证据：评估 jsonl / 汇总 / 曲线 / 探针
    ev = os.path.join(RUNS, "eval")
    for tag in os.listdir(ev):
        sd = os.path.join(ev, tag)
        if not os.path.isdir(sd):
            continue
        td = os.path.join(out, "evidence", tag)
        os.makedirs(td, exist_ok=True)
        for f in os.listdir(sd):
            if f.endswith(".jsonl") or f.endswith(".json"):
                shutil.copy2(os.path.join(sd, f), os.path.join(td, f))
    for extra in ("curves",):
        src = os.path.join(RUNS, extra)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(out, "evidence", extra), dirs_exist_ok=True)
    for probe in ("probe_P2BC1205.log", "eval_baseline.log", "eval_baseline_mission.log"):
        p = os.path.join(RUNS, probe)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, "evidence", probe))

    # 4) metadata.json
    cfg = json.load(open(os.path.join(RUNS, args.trial, "config.json"))) \
        if os.path.exists(os.path.join(RUNS, args.trial, "config.json")) else {}
    meta = dict(
        actor_file="actor_selected.pt",
        actor_sha256=sha,
        architecture=dict(obs_dim=355, n_actions=9, hidden=384, n_layers=2,
                          global_dim=8, note="与 deliverable/model/actor_best.pt 同结构"),
        contract_version=2,
        contract_note="UavOrder.to_vector 优先编码 resolve_order_target 的解析目标（rescuer_id 优先）",
        derived_from=dict(baseline_actor="deliverable/model/actor_best.pt",
                          baseline_sha256="e70e3f6169704b7b4027974049d3505a9113e2d6cfd38ceef93e46b460271168"),
        training=dict(trial_id=args.trial, reward_version=cfg.get("reward"),
                      roles=cfg.get("roles"), init=cfg.get("init"),
                      seed=cfg.get("seed"), actor_lr=cfg.get("actor_lr"),
                      ppo=dict(rollout=cfg.get("rollout"), clip=cfg.get("clip"),
                               ppo_epochs=cfg.get("ppo_epochs"),
                               entropy_start=0.02, entropy_final=0.01,
                               gamma=cfg.get("gamma"), gae_lambda=cfg.get("gae_lambda")),
                      sampler=dict(replan_every=cfg.get("replan_every"),
                                   role_randomized=cfg.get("roles") == "randomized",
                                   focus_probs=dict(S1=[0.35, 0.35, 0.15, 0.15],
                                                    S2=[0.30, 0.30, 0.20, 0.20]),
                                   stage_steps=dict(S1=cfg.get("stage_s1_steps"),
                                                    S2=cfg.get("stage_s2_steps"))),
                      scope=dict(num_uavs=[4], num_rescuers=[10, 14],
                                 grid="100x100", target_distance="8-90 格（课程分档）",
                                 note="训练分布 N=4；scarce(N=3) 通过参数共享外推，验收含 scarce")),
        eval_protocol=dict(split="quick 510000-510007 / final 520000+",
                           metrics="纯 Actor argmax，无 waypoint_assist/教师接管； relay=本机链路； explore=合法目标分层； patrol=每步解析环半径"),
        resumable_ckpt="ckpt_resumable.pt",
    )
    if args.meta and os.path.exists(args.meta):
        extra = json.load(open(args.meta))
        meta.update(extra)
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"[package] -> {out}")
    print(f"[package] actor_selected sha256={sha}")
    print(f"[package] trials={len(trial_table)}")


if __name__ == "__main__":
    main()
