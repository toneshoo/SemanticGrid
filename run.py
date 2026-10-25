"""
SemanticGrid. Run from the project root:

    python run.py train       # train the GNN surrogate on ANDES data
    python run.py verify      # L* + CEGIS: learn the FSM, check it against ANDES
    python run.py active      # active learning for hidden N-2 vulnerabilities
    python run.py n3          # exact ANDES ground truth for hidden N-3 triples (all CPU cores, resumable)
    python run.py weakpairs   # how many hidden N-3 triples sit around barely-safe pairs (needs n3 first)
    python run.py benchmark   # timings: ANDES vs GNN (CPU/GPU) vs FSM
    python run.py figures     # regenerate figures from saved results
"""
import importlib
import os
import sys
import warnings

COMMANDS = {
    "train": ("gnn", "main"),
    "verify": ("automata", "main"),
    "active": ("active_learning", "main"),
    "n3": ("n3", "main"),
    "weakpairs": ("n3", "weak_pairs"),
    "benchmark": ("analysis", "benchmark"),
    "figures": ("analysis", "figures"),
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    sys.dont_write_bytecode = True
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    sys.path.insert(0, os.path.join(root, "code"))
    warnings.filterwarnings("ignore")
    module, function = COMMANDS[sys.argv[1]]
    getattr(importlib.import_module(module), function)()
