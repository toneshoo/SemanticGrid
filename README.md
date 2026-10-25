# SemanticGrid

Automated vulnerability discovery in power grids using semantic automata learning and graph neural networks.

A graph neural network (GNN) trained on AC power flow acts as the membership oracle of the $L^*$ automata learning
algorithm. A counterexample-guided loop (CEGIS) checks every learned automaton against the physics simulator and
retrains the GNN on its mistakes. The result is a small, readable Moore machine of the grid's N-1 and N-2 line
outages, checked exhaustively against AC power flow.

Code and results for the paper *Automated Vulnerability Discovery in Power Grids using Semantic Automata Learning and
Graph Neural Networks* (M. Alsulami).

## Results (IEEE 118-bus system, 186 lines)

| | |
|---|---|
| GNN surrogate, held-out test set (357 cases) | 98.6% state accuracy, 0.0008 p.u. voltage error |
| Learned automaton | 115 states |
| Agreement with AC power flow, all 34,596 N-1 and ordered N-2 sequences | 98.5% |
| Dangerous N-2 pairs (3,490 of 17,205) | 98.6% recall, 95.1% precision |
| Hidden pairs (each line safe alone, dangerous together) | 146 (59 islanding, 87 voltage-driven) |
| Active learning, voltage-driven hidden pairs | ~80% found with ~1,200 physics solves (9% of brute force) |
| Automaton lookup vs Newton-Raphson | 0.00026 ms vs 6.10 ms per contingency |
| N-3, exact enumeration | 740,458 candidate triples, 372 hidden (0.05%) |

Semantic states come from the lowest bus voltage after the outage: NORMAL (>= 0.94 p.u.), ALERT (0.90-0.94),
EMERGENCY (0.85-0.90), COLLAPSE (< 0.85, islanded or diverged). The automaton covers up to two outages.

Known limitations: the CEGIS loop does not converge to zero disagreements; the automaton misses the two most severe
voltage-driven hidden pairs (L61+L67, L166+L168); no single corrective line trip restores NORMAL on this grid; and
results vary by about half a percentage point between GPU and CPU runs.

## Layout

| File | Job |
|---|---|
| `run.py` | Entry point for every step |
| `code/config.py` | Paths, seed, the four states and their voltage thresholds |
| `code/grid.py` | ANDES AC power flow, result cache, topology check, and the graph the GNN reads |
| `code/gnn.py` | Training data, GNN model and training, and the GNN as a fast oracle |
| `code/automata.py` | System under learning, and the $L^*$ + CEGIS loop checked against ANDES |
| `code/active_learning.py` | Active learning for hidden N-2 pairs |
| `code/n3.py` | Exact N-3 enumeration (all CPU cores, resumable) and the weak-pair analysis |
| `code/analysis.py` | Timing benchmark and figures |
| `code/reporting.py` | Console output |
| `models/` | Trained GNN, and the GNN after CEGIS fine-tuning |
| `results/` | All results used in the paper |

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt

python run.py train       # train the GNN on ANDES data
python run.py verify      # L* + CEGIS, exhaustive check against ANDES
python run.py active      # active learning for hidden N-2 pairs
python run.py benchmark   # timings
python run.py n3          # exact N-3 enumeration (about 25 min on 16 threads)
python run.py weakpairs   # hidden triples around barely-safe pairs
python run.py figures     # figures from saved results
```

ANDES results are cached in `data/andes_cache.json` (created on first run). Published results were produced on an
NVIDIA RTX 5070 Ti; CPU runs give slightly different but similar numbers.

Software: Python 3.11, PyTorch 2.11, PyTorch Geometric 2.8, ANDES 2.0, AALpy 1.6.
