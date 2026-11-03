"""distributed_allocator_v1: two synchronous broadcasts, deterministic local plans."""
import copy
import hashlib
import json
import math

VERSION = "distributed_allocator_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_all(snapshot, all_actions):
    ids = sorted(u["id"] for u in snapshot["uavs"])
    if snapshot["version"] != VERSION or set(map(int, all_actions)) != set(ids):
        raise ValueError("Incomplete/inconsistent synchronous broadcast")
    actions = {int(k): int(v) for k, v in all_actions.items()}
    if any(a not in (0, 1, 2) for a in actions.values()):
        raise ValueError("Invalid three-action message")
    people = sorted(snapshot["persons"], key=lambda p: (p["covered"], -p["aoi"], p["id"]))
    regions = snapshot["regions"]
    used_e, used_r, plans = set(), set(), {}
    by_id = {u["id"]: u for u in snapshot["uavs"]}
    # Reserve ALL valid old targets first, before assigning any new targets.
    for uid in ids:
        u, a = by_id[uid], actions[uid]
        if a != u["task"] or not u["valid"]:
            continue
        old = u["order"]
        if a == 0 and old.get("target") is not None:
            rid = min(range(25), key=lambda j: max(abs(regions[j]["center"][d] - old["target"][d]) for d in (0, 1)))
            if rid not in used_e:
                used_e.add(rid)
                plans[uid] = copy.deepcopy(old)
        elif a == 1 and old.get("rescuer_id") in {p["id"] for p in people}:
            rid = old["rescuer_id"]
            if rid not in used_r:
                used_r.add(rid)
                plans[uid] = copy.deepcopy(old)
    for uid in ids:
        if uid in plans:
            continue
        u, a = by_id[uid], actions[uid]
        if a == 2:
            plans[uid] = {"task": "charge"}
        elif a == 1:
            candidates = [p for p in people if p["id"] not in used_r] or people
            plans[uid] = {"task": "relay"}
            if candidates:
                rid = candidates[0]["id"]
                plans[uid]["rescuer_id"] = rid
                used_r.add(rid)
        else:
            candidates = [r for r in regions if r["id"] not in used_e] or regions
            if max(r["unknown"] for r in regions) > 0:
                r = min(candidates, key=lambda r: (-r["unknown"],
                        max(abs(r["center"][d] - u["pos"][d]) for d in (0, 1)), r["id"]))
            else:
                fire = [r for r in candidates if r["fire"] > 0]
                pool = fire or candidates
                r = min(pool, key=lambda r: ((r["id"] - snapshot["t"] // 20) % 25, r["id"]))
            plans[uid] = {"task": "explore", "mode": "explore", "target": r["center"]}
            used_e.add(r["id"])
    return plans


def resolve_target(local_snapshot, all_actions, self_id):
    return resolve_all(local_snapshot, all_actions)[self_id]


def baseline(snapshot, kind="h3", threshold=260, relay_mode="one", train_seed=0):
    uavs = sorted(snapshot["uavs"], key=lambda u: u["id"])
    n = len(uavs)
    if kind == "random":
        import numpy as np
        return [int(np.random.default_rng(np.random.SeedSequence(
            [train_seed, snapshot["seed"], snapshot["t"], u["id"]])).integers(3)) for u in uavs]
    if kind == "fixed":
        return [0 if i < math.ceil(n * .5) else 1 if i < math.ceil(n * .75) else 2 for i in range(n)]
    actions = {u["id"]: 0 for u in uavs}
    rest = []
    for u in sorted(uavs, key=lambda u: (u["energy"], u["id"])):
        if u["energy"] < threshold:
            actions[u["id"]] = 2
        else:
            rest.append(u)
    people = sorted(snapshot["persons"], key=lambda p: (p["covered"], -p["aoi"], p["id"]))
    for p in people[:1 if relay_mode == "one" else max(1, math.ceil(n / 4))]:
        if rest:
            u = min(rest, key=lambda u: (max(abs(u["pos"][d] - p["pos"][d]) for d in (0, 1)), u["id"]))
            actions[u["id"]] = 1
            rest.remove(u)
    return [actions[u["id"]] for u in uavs]


def h3_local(local_snapshot, self_id, threshold=300, relay_mode="one"):
    """Each vehicle independently repeats H3 on the identical public snapshot."""
    ids = sorted(u["id"] for u in local_snapshot["uavs"])
    actions = baseline(local_snapshot, "h3", threshold, relay_mode)
    return actions[ids.index(self_id)]
