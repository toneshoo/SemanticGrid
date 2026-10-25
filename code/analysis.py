import json
import math
import os
import random
import statistics
import time

import torch

from config import GRID_CASE, RESULTS_DIR
from gnn import SemanticGNNOracle, load_model
from grid import GridSimulator
from reporting import print_benchmark_summary

N_SAMPLES = 200
GRID_NAME = os.path.splitext(os.path.basename(GRID_CASE))[0]


def timed(fn, items, repeats=1):
    times = []
    for item in items:
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(item)
        times.append((time.perf_counter() - t0) / repeats * 1000)
    return statistics.median(times)


def benchmark():
    torch.set_num_threads(1)
    sim = GridSimulator(GRID_CASE)
    n = sim.num_lines
    rng = random.Random(0)
    contingencies = [sorted(rng.sample(range(n), rng.choice([1, 2, 3]))) for _ in range(N_SAMPLES)]
    nr_ms = timed(lambda c: sim.solve(c, use_cache=False), contingencies)

    oracle = SemanticGNNOracle(load_model("cpu"), sim, device="cpu")
    for c in contingencies[:20]:
        oracle.predict(c)
    gnn_ms = timed(oracle.predict, contingencies)

    with open(os.path.join(RESULTS_DIR, f"{GRID_NAME}_fsm.json"), encoding="utf-8") as f:
        fsm = json.load(f)

    def lookup(seq):
        s = fsm["initial"]
        for a in seq:
            s = fsm["states"][s]["transitions"][str(a)]
        return fsm["states"][s]["output"]

    fsm_ms = timed(lookup, contingencies, repeats=1000)

    gpu = {}
    if torch.cuda.is_available():
        gpu_oracle = SemanticGNNOracle(load_model("cuda"), sim, device="cuda")

        def sync_predict(c):
            gpu_oracle.predict(c)
            torch.cuda.synchronize()

        for c in contingencies[:20]:
            sync_predict(c)
        gpu["single_ms"] = timed(sync_predict, contingencies)
        all_pairs = [[a, b] for a in range(n) for b in range(a + 1, n)]
        for bs in (256, 1024, 4096):
            gpu_oracle.predict_batch(all_pairs[:bs])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(0, len(all_pairs), bs):
                gpu_oracle.predict_batch(all_pairs[i:i + bs])
            torch.cuda.synchronize()
            gpu[f"batch{bs}_ms_per_contingency"] = (time.perf_counter() - t0) * 1000 / len(all_pairs)
        gpu["device"] = torch.cuda.get_device_name(0)
        gpu["all_n2_pairs_seconds"] = min(v for k, v in gpu.items() if k.startswith("batch")) * len(all_pairs) / 1000

    counts = {k: math.comb(n, k) for k in (1, 2, 3)}
    results = {
        "grid": GRID_NAME, "lines": n, "samples": N_SAMPLES,
        "median_ms": {"newton_raphson_andes": nr_ms, "gnn_inference": gnn_ms, "fsm_lookup": fsm_ms},
        "speedup_vs_nr": {"gnn": nr_ms / gnn_ms, "fsm": nr_ms / fsm_ms},
        "gpu_gnn": gpu,
        "contingency_counts": counts,
        "estimated_full_sweep_hours_nr": {f"N-{k}": c * nr_ms / 1000 / 3600 for k, c in counts.items()},
        "machine_note": "ANDES / CPU GNN / FSM: CPU, single thread; GPU GNN: CUDA; Python 3.11",
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    gpu_batched_ms = gpu.get("batch4096_ms_per_contingency")
    print_benchmark_summary(GRID_NAME, N_SAMPLES, nr_ms, gnn_ms, gpu_batched_ms, fsm_ms,
                            results["estimated_full_sweep_hours_nr"]["N-3"])


def draw_fsm(fsm_path, output_jpg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with open(fsm_path, encoding="utf-8") as f:
        fsm = json.load(f)
    states, start, alphabet = fsm["states"], fsm["initial"], fsm["alphabet"]
    colors = {"NORMAL": "#4c9be8", "ALERT": "#f2b134", "EMERGENCY": "#ef7d3c", "COLLAPSE": "#d64545"}
    first = {}
    for a in alphabet:
        first.setdefault(states[start]["transitions"][str(a)], []).append(a)

    def danger(s):
        return {a for a in alphabet if states[states[s]["transitions"][str(a)]]["output"] != "NORMAL"}

    base = danger(start)

    def group(s):
        if states[s]["output"] != "NORMAL":
            return 0
        d = danger(s)
        return 1 if d > base else 2 if d != base else 3

    titles = ["Visibly damaged", "Looks NORMAL, weakened\n(more lines now dangerous)",
              "Looks NORMAL, different\nweak spots", "Looks NORMAL, same weak\nspots as the intact grid"]
    rank = {o: i for i, o in enumerate(colors)}
    sinks = [s for s in states if all(t == s for t in states[s]["transitions"].values())]
    finals = {o: next(s for s in sinks if states[s]["output"] == o) for o in colors}
    order = sorted((s for s in first if s not in finals.values()), key=lambda s: (group(s), -len(first[s])))

    fig, ax = plt.subplots(figsize=(16, 22), dpi=110)
    pos, x_cols, row_h = {}, [3.0, 4.6, 6.2], 0.5
    y = 0.0
    for g in range(4):
        members = [s for s in order if group(s) == g]
        ax.text(x_cols[0] - 0.2, y + 0.45, titles[g].replace("\n", " "), fontsize=12, fontweight="bold", va="bottom")
        for i, s in enumerate(members):
            pos[s] = (x_cols[i % 3], y - (i // 3) * row_h)
        y -= (len(members) + 2) // 3 * row_h + 1.6
    pos[start] = (0.0, y / 2)
    for i, o in enumerate(colors):
        pos[finals[o]] = (9.5, y / 2 + (1.5 - i) * 4)

    for s in order:
        ax.plot(*zip(pos[start], pos[s]), color="#bbbbbb", lw=0.4 + 0.15 * len(first[s]), zorder=1)
        counts = {}
        for a in alphabet:
            o = states[states[s]["transitions"][str(a)]]["output"]
            counts[o] = counts.get(o, 0) + 1
        for o, n in counts.items():
            if rank[o] > rank[states[s]["output"]]:
                ax.plot(*zip(pos[s], pos[finals[o]]), color=colors[o], lw=0.3 + n / 8, alpha=0.35, zorder=1)
    for o, s in finals.items():
        if s in first:
            ax.plot(*zip(pos[start], pos[s]), color=colors[o], lw=0.4 + 0.15 * len(first[s]), alpha=0.6, zorder=1)
            mid = ((pos[start][0] + pos[s][0]) / 2, (pos[start][1] + pos[s][1]) / 2 - 0.4)
            ax.text(*mid, f"{len(first[s])} lines: {o} after one trip", fontsize=9, color=colors[o], ha="center")

    for s in order:
        lines = first[s]
        label = ", ".join(f"L{a}" for a in lines) if len(lines) <= 2 else f"{len(lines)} lines"
        ax.scatter(*pos[s], s=60 + 25 * len(lines), color=colors[states[s]["output"]], edgecolor="black", lw=0.6,
                   zorder=3)
        ax.text(pos[s][0] + 0.18, pos[s][1], label, fontsize=8, va="center", zorder=4)
    ax.scatter(*pos[start], s=3800, color=colors["NORMAL"], edgecolor="black", lw=1.5, zorder=3)
    ax.text(*pos[start], "START\nnothing\ntripped", ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)
    for o, s in finals.items():
        ax.scatter(*pos[s], s=4200, color=colors[o], edgecolor="black", lw=1.5, zorder=3)
        ax.text(*pos[s], f"{o}\nafter 2 trips", ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)

    ax.text(0.0, 1.6, f"Learned FSM, {GRID_NAME}: {len(states)} states. Each arrow is one line trip.\n"
            "Trip 1: START to one of the middle states (size = number of lines that lead there). "
            "Trip 2: to an outcome on the right\n(only second trips that make the grid worse are drawn; "
            "line width = number of lines).", fontsize=13, va="bottom")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, markeredgecolor="black", ms=10, label=o)
                       for o, c in colors.items()], loc="lower left", fontsize=11, frameon=False, title="Grid state")
    ax.set_xlim(-1.2, 10.8)
    ax.set_ylim(y - 0.5, 3.2)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_jpg, pil_kwargs={"quality": 92})
    plt.close(fig)


def plot_active_learning(results, output_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"random": "#8a8a8a", "uncertainty": "#2a7ab9", "danger_uncertainty": "#d1495b"}
    seeds = results["config"]["seeds"]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for strategy, color in colors.items():
        runs = [results["runs"][f"{strategy}/{s}"] for s in seeds if f"{strategy}/{s}" in results["runs"]]
        if not runs:
            continue
        x = [sum(r[i]["physics_calls_total"] for r in runs) / len(runs) for i in range(len(runs[0]))]
        ys = np.array([[h["hidden_recall"] * 100 for h in r] for r in runs])
        ax.plot(x, ys.mean(0), marker="o", ms=3, color=color, label=strategy.replace("_", " + "))
        ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.15)
    kind = results.get("kind", "pairs")
    ax.axvline(results["pool_size"], color="black", ls="--", lw=1)
    ax.text(results["pool_size"], 3, " brute force", fontsize=8, ha="left")
    ax.set_xlabel("ANDES power-flow solves (labels + confirming GNN flags)")
    ax.set_ylabel(f"Hidden N-{2 if kind == 'pairs' else 3} {kind} found (%)")
    ax.set_ylim(0, 102)
    if kind == "pairs":
        ax.set_xlim(0, results["pool_size"] * 1.08)
    else:
        ax.set_xscale("log")
        ax.set_xlim(100, results["pool_size"] * 2)
    warm = ", warm start" if results.get("warm_start") else ""
    title = f"{GRID_NAME}: {results['hidden_pairs']} hidden {kind} among {results['pool_size']:,} candidates{warm}"
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_png)


def figures():
    draw_fsm(os.path.join(RESULTS_DIR, f"{GRID_NAME}_fsm.json"), os.path.join(RESULTS_DIR, f"{GRID_NAME}_fsm.jpg"))
    al_json = os.path.join(RESULTS_DIR, "active_learning.json")
    if os.path.exists(al_json):
        with open(al_json, encoding="utf-8") as f:
            plot_active_learning(json.load(f), os.path.join(RESULTS_DIR, "active_learning.png"))
