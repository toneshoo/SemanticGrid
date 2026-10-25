GRID_CASE = "grids/case118.m"
WEIGHTS = "models/gnn_oracle_weights.pth"
WEIGHTS_CEGIS = "models/gnn_oracle_weights_cegis.pth"
CACHE = "data/andes_cache.json"
RESULTS_DIR = "results"
SEED = 0

STATES = ["NORMAL", "ALERT", "EMERGENCY", "COLLAPSE"]
V_NORMAL, V_ALERT, V_EMERGENCY = 0.94, 0.90, 0.85


def classify(min_v, collapsed=False):
    if collapsed or min_v is None:
        return "COLLAPSE"
    if min_v >= V_NORMAL:
        return "NORMAL"
    if min_v >= V_ALERT:
        return "ALERT"
    if min_v >= V_EMERGENCY:
        return "EMERGENCY"
    return "COLLAPSE"
