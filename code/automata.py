import json
import os
import random
import sys

import torch
from aalpy.base import SUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWordEqOracle
from tqdm import tqdm

from config import CACHE, GRID_CASE, RESULTS_DIR, SEED, STATES, WEIGHTS, WEIGHTS_CEGIS, classify
from gnn import SemanticGNNOracle, build_dataset, evaluate, load_model, split, train
from grid import GridSimulator
from reporting import print_cegis_round, print_verification_summary

MAX_CEGIS_ITERATIONS = 6
MAX_TRIPS = 2
CEGIS_N2_SAMPLE = 2000
SEED_CEGIS = 1


class SemanticGridSUL(SUL):
    def __init__(self, backend, max_trips=MAX_TRIPS, progress=None):
        super().__init__()
        self.backend, self.max_trips, self.current_trips = backend, max_trips, []
        self.bar = tqdm(desc=progress, unit=" queries", file=sys.__stderr__, dynamic_ncols=True) if progress else None

    def pre(self):
        self.current_trips = []
        if self.bar is not None:
            self.bar.update(1)

    def step(self, letter):
        if letter is not None and letter not in self.current_trips and len(self.current_trips) < self.max_trips:
            self.current_trips.append(letter)
        return self.backend.semantic_state(self.current_trips)

    def post(self):
        pass


class AndesBackend:
    def __init__(self, sim):
        self.sim = sim

    def semantic_state(self, tripped_lines):
        res = self.sim.solve(tripped_lines)
        return classify(min(res["v"])) if res["converged"] else "COLLAPSE"


def fsm_output(fsm, sequence):
    state = fsm.initial_state
    for letter in sequence:
        state = state.transitions[letter]
    return state.output


def check_against_andes(fsm, suite, truth):
    mismatches = []
    for seq in suite:
        expected, got = truth.semantic_state(seq), fsm_output(fsm, seq)
        if got != expected:
            mismatches.append({"trips": list(seq), "fsm": got, "andes": expected})
    return 1 - len(mismatches) / max(len(suite), 1), mismatches


def fsm_stats(fsm, alphabet):
    total = len(fsm.states) * len(alphabet)
    self_loops = sum(1 for s in fsm.states for a in alphabet if s.transitions[a] is s)
    counts = {}
    for s in fsm.states:
        counts[s.output] = counts.get(s.output, 0) + 1
    return {"states": len(fsm.states), "transitions": total, "self_loops": self_loops,
            "non_self_loop_transitions": total - self_loops, "states_by_output": counts}


def min_v(sim, trips):
    res = sim.solve(trips)
    return min(res["v"]) if res["converged"] else None


def analyse(fsm, sim, truth, alphabet):
    severity = {s: i for i, s in enumerate(STATES)}
    vulnerabilities = [{"line": a, "label": sim.line_label(a), "fsm": fsm_output(fsm, [a]),
                        "andes": truth.semantic_state([a]), "andes_min_v": min_v(sim, [a])}
                       for a in alphabet if fsm_output(fsm, [a]) != "NORMAL"]
    vulnerabilities.sort(key=lambda v: (-severity[v["andes"]], v["andes_min_v"] or 0))
    remediations = [{"after_line": v["line"], "trip_line": b, "trip_label": sim.line_label(b), "fsm": "NORMAL",
                     "andes": truth.semantic_state([v["line"], b]), "andes_min_v": min_v(sim, [v["line"], b])}
                    for v in vulnerabilities if v["andes"] != "NORMAL"
                    for b in alphabet if b != v["line"] and fsm_output(fsm, [v["line"], b]) == "NORMAL"]
    return vulnerabilities, remediations


def detection(fsm, sim, truth, alphabet):
    n1_safe = {a for a in alphabet if truth.semantic_state([a]) == "NORMAL"}
    tp = fp = fn = 0
    hidden = []
    for a in alphabet:
        for b in alphabet[a + 1:]:
            actual = truth.semantic_state([a, b])
            outputs = [fsm_output(fsm, [a, b]), fsm_output(fsm, [b, a])]
            flagged, dangerous = any(o != "NORMAL" for o in outputs), actual != "NORMAL"
            tp += dangerous and flagged
            fp += flagged and not dangerous
            fn += dangerous and not flagged
            if dangerous and a in n1_safe and b in n1_safe:
                hidden.append({"lines": [a, b], "labels": [sim.line_label(a), sim.line_label(b)], "andes": actual,
                               "andes_min_v": min_v(sim, [a, b]), "fsm": outputs, "flagged": flagged})
    severity = {s: i for i, s in enumerate(STATES)}
    hidden.sort(key=lambda h: (-severity[h["andes"]], h["andes_min_v"] if h["andes_min_v"] is not None else 0))
    caught = sum(h["flagged"] for h in hidden)
    return {"n2_dangerous_pairs": tp + fn, "n2_recall": tp / max(tp + fn, 1), "n2_precision": tp / max(tp + fp, 1),
            "n2_false_alarms": fp, "hidden_pairs": len(hidden), "hidden_pairs_flagged": caught,
            "hidden_recall": caught / max(len(hidden), 1),
            "hidden_by_class": {s: sum(h["andes"] == s for h in hidden) for s in STATES[1:]}}, hidden


def export_fsm(fsm, alphabet, json_path):
    states = {s.state_id: {"output": s.output, "prefix": [int(p) for p in s.prefix],
                           "transitions": {str(a): s.transitions[a].state_id for a in alphabet}} for s in fsm.states}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"initial": fsm.initial_state.state_id, "alphabet": alphabet, "states": states}, f)


def main():
    if not os.path.exists(WEIGHTS):
        sys.exit(f"{WEIGHTS} missing: run `python run.py train` first")
    grid_name = os.path.splitext(os.path.basename(GRID_CASE))[0]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = GridSimulator(GRID_CASE, cache_path=CACHE)
    truth = AndesBackend(sim)
    alphabet = list(range(sim.num_lines))
    model = load_model(device)

    train_data, _, test_data = split(build_dataset(sim))
    n1_suite = [[a] for a in alphabet]
    n2_all = [[a, b] for a in alphabet for b in alphabet if a != b]
    for i, seq in enumerate(n2_all, 1):
        sim.solve(seq)
        if i % 5000 == 0:
            sim.save_cache()
    sim.save_cache()
    cegis_suite = n1_suite + random.Random(SEED_CEGIS).sample(n2_all, CEGIS_N2_SAMPLE)
    cegis_keys = {tuple(s) for s in cegis_suite}
    holdout_suite = [s for s in n2_all if tuple(s) not in cegis_keys]

    history, fsm = [], None
    for iteration in range(1, MAX_CEGIS_ITERATIONS + 1):
        sul = SemanticGridSUL(SemanticGNNOracle(model, sim, device=device), progress=f"L* round {iteration}")
        eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=5000, min_walk_len=1, max_walk_len=MAX_TRIPS + 1)
        fsm = run_Lstar(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type="moore", print_level=0)
        sul.bar.close()
        agreement, mismatches = check_against_andes(fsm, cegis_suite, truth)
        history.append({"iteration": iteration, "states": len(fsm.states), "cegis_agreement": agreement,
                        "counterexamples": len(mismatches)})
        print_cegis_round(iteration, len(fsm.states), agreement, len(mismatches))
        if not mismatches or iteration == MAX_CEGIS_ITERATIONS:
            break
        counter_examples = [sim.to_data(m["trips"]) for m in mismatches]
        train_data = train_data + counter_examples
        model = train(model, train_data + counter_examples * 3, epochs=40, lr=1e-3, device=device)
        torch.save(model.state_dict(), WEIGHTS_CEGIS)

    n1_agreement, n1_mismatches = check_against_andes(fsm, n1_suite, truth)
    cegis_agreement, _ = check_against_andes(fsm, cegis_suite, truth)
    holdout_agreement, holdout_mismatches = check_against_andes(fsm, holdout_suite, truth)
    exhaustive_agreement, _ = check_against_andes(fsm, n1_suite + n2_all, truth)
    vulnerabilities, remediations = analyse(fsm, sim, truth, alphabet)
    det, hidden = detection(fsm, sim, truth, alphabet)
    export_fsm(fsm, alphabet, os.path.join(RESULTS_DIR, f"{grid_name}_fsm.json"))
    torch.save(model.state_dict(), WEIGHTS_CEGIS)

    report = {
        "grid": grid_name, "lines": sim.num_lines, "cegis_history": history, "fsm": fsm_stats(fsm, alphabet),
        "agreement_with_andes": {"n1_exhaustive": n1_agreement, "cegis_suite": cegis_agreement,
                                 "holdout_n2": holdout_agreement, "exhaustive_n1_n2": exhaustive_agreement,
                                 "max_trips": MAX_TRIPS, "holdout_size": len(holdout_suite)},
        "remaining_mismatches": {"n1": n1_mismatches, "holdout_count": len(holdout_mismatches),
                                 "holdout_first_200": holdout_mismatches[:200]},
        "gnn_test": evaluate(model, test_data, device),
        "vulnerabilities": vulnerabilities, "remediations": remediations, "detection": det, "hidden_pairs": hidden,
    }
    with open(os.path.join(RESULTS_DIR, f"{grid_name}_cegis_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    confirmed = sum(r["andes"] == "NORMAL" for r in remediations)
    print_verification_summary(len(fsm.states), n1_agreement, holdout_agreement, exhaustive_agreement,
                               det, confirmed, len(remediations))
