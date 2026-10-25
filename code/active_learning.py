import json
import math
import os
import random
import time

import torch
from tqdm import tqdm

from analysis import plot_active_learning
from automata import AndesBackend
from config import CACHE, GRID_CASE, RESULTS_DIR, STATES, WEIGHTS
from gnn import PowerGridGNN, SemanticGNNOracle, train
from grid import GridSimulator
from reporting import print_active_learning_pool_summary, print_active_learning_run_summary

STRATEGIES = ["random", "uncertainty", "danger_uncertainty"]
SEEDS = [0, 1, 2]
INIT_N2 = 200
BUDGET = 200
ROUNDS = 10
INIT_EPOCHS, ROUND_EPOCHS = 200, 60


def build_pool(sim, truth):
    n1_safe = [a for a in range(sim.num_lines) if truth.semantic_state([a]) == "NORMAL"]
    pairs = [[a, b] for i, a in enumerate(n1_safe) for b in n1_safe[i + 1:]]
    pool = [p for p in pairs if sim.is_connected(p)]
    return pool, [truth.semantic_state(p) for p in pool], len(n1_safe), len(pairs) - len(pool)


def score_pool(model, sim, pool, device):
    oracle = SemanticGNNOracle(model, sim, device=device)
    states, probs = [], []
    for i in range(0, len(pool), 4096):
        _, s, p = oracle.predict_batch(pool[i:i + 4096], return_probs=True)
        states += s
        probs.append(p)
    return states, torch.cat(probs)


def acquire(strategy, probs, unlabeled, rng, k):
    if strategy == "random":
        return rng.sample(unlabeled, k)
    p = probs[unlabeled]
    entropy = -(p * p.clamp_min(1e-12).log()).sum(1) / math.log(len(STATES))
    score = entropy if strategy == "uncertainty" else (1 - p[:, STATES.index("NORMAL")]) + 0.5 * entropy
    return [unlabeled[i] for i in torch.topk(score, k).indices.tolist()]


def run(strategy, seed, sim, pool, labels, n1_data, device):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    dangerous = {i for i, l in enumerate(labels) if l != "NORMAL"}
    labeled = set(rng.sample(range(len(pool)), INIT_N2))
    model = PowerGridGNN().to(device)
    model.set_normalization(sim.node_features.to(device))
    history = []
    bar = tqdm(total=ROUNDS + 1, desc=f"  {strategy} seed {seed}", unit="round", leave=False, dynamic_ncols=True)
    for r in range(ROUNDS + 1):
        data = n1_data + [sim.to_data(pool[i]) for i in labeled]
        train(model, data, epochs=INIT_EPOCHS if r == 0 else ROUND_EPOCHS, lr=3e-3 if r == 0 else 1e-3, device=device)
        states, probs = score_pool(model, sim, pool, device)
        found_by_physics = dangerous & labeled
        flagged = {i for i, s in enumerate(states) if s != "NORMAL" and i not in labeled}
        found = found_by_physics | (flagged & dangerous)
        history.append({"round": r, "physics_calls_labels": len(labeled),
                        "physics_calls_total": len(labeled) + len(flagged), "hidden_found": len(found),
                        "hidden_recall": len(found) / len(dangerous), "found_by_physics_labels": len(found_by_physics),
                        "gnn_flags": len(flagged),
                        "gnn_flag_precision": len(flagged & dangerous) / max(len(flagged), 1)})
        bar.set_postfix(found=f"{len(found)}/{len(dangerous)}", solves=len(labeled) + len(flagged))
        bar.update(1)
        if r == ROUNDS:
            break
        unlabeled = [i for i in range(len(pool)) if i not in labeled]
        labeled |= set(acquire(strategy, probs, unlabeled, rng, BUDGET))
    bar.close()
    h = history[-1]
    print_active_learning_run_summary(strategy, seed, False, h["hidden_found"], len(dangerous), "pairs",
                                      h["hidden_recall"], h["physics_calls_total"])
    return history, found


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sim = GridSimulator(GRID_CASE, cache_path=CACHE)
    pool, labels, n_safe, islanding = build_pool(sim, AndesBackend(sim))
    n_hidden = sum(l != "NORMAL" for l in labels)
    print_active_learning_pool_summary(len(pool), n_hidden, "pairs", False, WEIGHTS)
    n1_data = [sim.to_data([])] + [sim.to_data([a]) for a in range(sim.num_lines)]
    results = {"pool_size": len(pool), "hidden_pairs": n_hidden, "islanding_removed": islanding,
               "config": {"init_n2": INIT_N2, "budget": BUDGET, "rounds": ROUNDS, "seeds": SEEDS}, "runs": {},
               "missed_hidden_pairs": {}}
    t0 = time.perf_counter()
    for strategy in STRATEGIES:
        for seed in SEEDS:
            history, found = run(strategy, seed, sim, pool, labels, n1_data, device)
            results["runs"][f"{strategy}/{seed}"] = history
            results["missed_hidden_pairs"][f"{strategy}/{seed}"] = [
                {"lines": pool[i], "labels": [sim.line_label(x) for x in pool[i]], "andes": labels[i],
                 "min_v": min(sim.solve(pool[i])["v"])}
                for i, l in enumerate(labels) if l != "NORMAL" and i not in found]
    results["seconds"] = time.perf_counter() - t0
    with open(os.path.join(RESULTS_DIR, "active_learning.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    plot_active_learning(results, os.path.join(RESULTS_DIR, "active_learning.png"))
