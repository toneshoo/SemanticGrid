import itertools
import json
import os
import time
from multiprocessing import Pool

from automata import AndesBackend
from config import CACHE, GRID_CASE, RESULTS_DIR, STATES, classify
from grid import GridSimulator
from reporting import print_n3_candidates, print_n3_progress, print_n3_summary

GRID_NAME = os.path.splitext(os.path.basename(GRID_CASE))[0]
OUT = os.path.join(RESULTS_DIR, f"{GRID_NAME}_n3_ground_truth.json")
SAVE_EVERY = 20000
_sim = None


def candidates(sim, truth):
    safe = [a for a in range(sim.num_lines) if truth.semantic_state([a]) == "NORMAL"]
    safe_pairs = {(a, b) for a, b in itertools.combinations(safe, 2)
                  if sim.is_connected([a, b]) and truth.semantic_state([a, b]) == "NORMAL"}
    triples, islanding = [], 0
    for a, b, c in itertools.combinations(safe, 3):
        if (a, b) in safe_pairs and (a, c) in safe_pairs and (b, c) in safe_pairs:
            if sim.is_connected([a, b, c]):
                triples.append((a, b, c))
            else:
                islanding += 1
    return triples, len(safe), len(safe_pairs), islanding


def _init_worker():
    global _sim
    _sim = GridSimulator(GRID_CASE)


def _solve(triple):
    res = _sim.solve(list(triple))
    return GridSimulator.key(triple), res["status"], min(res["v"]) if res["converged"] else None


def load_results():
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    return {"grid": GRID_NAME, "solved": {}}


def save_results(results):
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f)
    os.replace(tmp, OUT)


def summarise(results):
    states = [classify(v, collapsed=s != "ok") for s, v in results["solved"].values()]
    return {s: states.count(s) for s in STATES}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sim = GridSimulator(GRID_CASE, cache_path=CACHE)
    triples, n_safe, n_safe_pairs, islanding = candidates(sim, AndesBackend(sim))
    results = load_results()
    results.update(n1_safe_lines=n_safe, n2_safe_pairs=n_safe_pairs, candidate_triples=len(triples),
                   islanding_removed=islanding)
    todo = [t for t in triples if GridSimulator.key(t) not in results["solved"]]
    print_n3_candidates(GRID_NAME, len(triples), islanding, len(todo), os.cpu_count())
    t0, done = time.perf_counter(), 0
    with Pool(os.cpu_count(), initializer=_init_worker) as pool:
        for key, status, v in pool.imap_unordered(_solve, todo, chunksize=64):
            results["solved"][key] = [status, v]
            done += 1
            if done % SAVE_EVERY == 0:
                save_results(results)
                rate = done / (time.perf_counter() - t0)
                print_n3_progress(done, len(todo), (len(todo) - done) / rate / 60)
    results["triples_by_state"] = summarise(results)
    save_results(results)
    print_n3_summary(results["triples_by_state"], OUT)


def weak_pairs(thresholds=(0.9405, 0.941, 0.942, 0.9425)):
    sim = GridSimulator(GRID_CASE, cache_path=CACHE)
    truth = load_results()["solved"]
    pair_v = {}
    hidden, weakest = [], []
    for k, (s, v) in truth.items():
        t = list(map(int, k.split(",")))
        w = []
        for a, b in itertools.combinations(t, 2):
            if (a, b) not in pair_v:
                res = sim.solve([a, b])
                pair_v[(a, b)] = min(res["v"]) if res["converged"] else 0.0
            w.append(pair_v[(a, b)])
        weakest.append(min(w))
        hidden.append(classify(v, collapsed=s != "ok") != "NORMAL")
    n_hidden = sum(hidden)
    print(f"{len(truth):,} candidate triples, {n_hidden} hidden")
    for thr in thresholds:
        sel = [h for h, w in zip(hidden, weakest) if w < thr]
        print(f"  weakest pair < {thr}: {len(sel):,} triples ({len(sel) / len(truth):.1%}) contain "
              f"{sum(sel)}/{n_hidden} hidden ({sum(sel) / n_hidden:.0%})")
