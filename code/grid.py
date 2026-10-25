import json
import os
import time
import warnings

import andes
import numpy as np
import torch
from torch_geometric.data import Data

from config import STATES, classify

warnings.filterwarnings("ignore")
andes.config_logger(stream_level=50)

ISLAND_V_FLOOR = 0.5


def resolve_case(grid_case):
    if os.path.exists(grid_case):
        return grid_case
    return andes.get_case(f"matpower/{os.path.basename(grid_case)}")


class GridSimulator:
    def __init__(self, grid_case, cache_path=None):
        self.case_file = resolve_case(grid_case)
        self._load()
        self.line_ids = list(self.ss.Line.idx.v)
        self.num_lines = len(self.line_ids)
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)
        base = self.solve([])
        if not base["converged"]:
            raise RuntimeError(f"Base case {self.case_file} does not converge")
        self.base_v = np.array(base["v"])
        self._build_graph()

    def _load(self):
        self.ss = andes.load(self.case_file, setup=True, no_output=True, default_config=True)

    def _build_graph(self):
        ss = self.ss
        self.bus_pos = {b: i for i, b in enumerate(ss.Bus.idx.v)}
        self.bus_names = [str(n) for n in ss.Bus.name.v]
        n = len(self.bus_pos)
        ss.PFlow.run()
        base_a = np.array(ss.Bus.a.v, dtype=float)
        p_load, q_load, p_gen, is_pv = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        for b, p, q in zip(ss.PQ.bus.v, ss.PQ.p0.v, ss.PQ.q0.v):
            p_load[self.bus_pos[b]] += p
            q_load[self.bus_pos[b]] += q
        for b, p in zip(ss.PV.bus.v, ss.PV.p0.v):
            p_gen[self.bus_pos[b]] += p
            is_pv[self.bus_pos[b]] = 1.0
        for b in ss.Slack.bus.v:
            is_pv[self.bus_pos[b]] = 1.0
        x = np.column_stack((self.base_v, base_a, p_load, q_load, p_gen, is_pv))
        self.node_features = torch.tensor(x, dtype=torch.float)
        self.line_from = [self.bus_pos[b] for b in ss.Line.bus1.v]
        self.line_to = [self.bus_pos[b] for b in ss.Line.bus2.v]
        rx = torch.tensor(np.column_stack((ss.Line.r.v, ss.Line.x.v)), dtype=torch.float)
        self.edge_index = torch.tensor([self.line_from + self.line_to, self.line_to + self.line_from], dtype=torch.long)
        self.edge_attr = torch.cat([rx, rx], dim=0)
        self.edge_line = torch.arange(self.num_lines).repeat(2)

    @staticmethod
    def key(tripped_lines):
        return ",".join(str(i) for i in sorted(set(tripped_lines)))

    def solve(self, tripped_lines, use_cache=True):
        k = self.key(tripped_lines)
        if use_cache and k in self.cache:
            return self.cache[k]
        trips = sorted(set(tripped_lines))
        for i in trips:
            self.ss.Line.alter("u", self.line_ids[i], 0)
        try:
            self.ss.PFlow.run()
            converged = bool(self.ss.PFlow.converged)
            v = np.array(self.ss.Bus.v.v, dtype=float)
            if not converged:
                result = {"converged": False, "v": None, "status": "diverged"}
            elif not np.all(np.isfinite(v)) or v.min() < ISLAND_V_FLOOR:
                result = {"converged": False, "v": None, "status": "islanded"}
            else:
                result = {"converged": True, "v": v.round(6).tolist(), "status": "ok"}
        except Exception:
            result = {"converged": False, "v": None, "status": "diverged"}
        finally:
            for i in trips:
                self.ss.Line.alter("u", self.line_ids[i], 1)
        if result["status"] != "ok":
            self._load()  # a failed solve leaves NaNs in ANDES that corrupt every later solve
        self.cache[k] = result
        return result

    def save_cache(self):
        if self.cache_path:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)

    def edge_mask(self, tripped_lines):
        mask = torch.ones(self.edge_index.shape[1], dtype=torch.bool)
        for i in set(tripped_lines):
            mask[self.edge_line == i] = False
        return mask

    def to_data(self, tripped_lines):
        res = self.solve(tripped_lines)
        mask = self.edge_mask(tripped_lines)
        if res["converged"]:
            y, collapse, label = torch.tensor(res["v"], dtype=torch.float), 0.0, classify(min(res["v"]))
        else:
            y, collapse, label = torch.full((self.node_features.shape[0],), float("nan")), 1.0, "COLLAPSE"
        return Data(x=self.node_features, edge_index=self.edge_index[:, mask], edge_attr=self.edge_attr[mask], y=y,
                    collapse=torch.tensor([collapse]), label=torch.tensor([STATES.index(label)]),
                    trips=self.key(tripped_lines))

    def is_connected(self, tripped_lines):
        out = set(tripped_lines)
        adj = [[] for _ in self.bus_names]
        for i in range(self.num_lines):
            if i not in out:
                adj[self.line_from[i]].append(self.line_to[i])
                adj[self.line_to[i]].append(self.line_from[i])
        seen, stack = {0}, [0]
        while stack:
            for w in adj[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return len(seen) == len(self.bus_names)

    def line_label(self, i):
        return f"L{i} ({self.bus_names[self.line_from[i]]} - {self.bus_names[self.line_to[i]]})"

    def time_solve(self, tripped_lines, repeats=1):
        t0 = time.perf_counter()
        for _ in range(repeats):
            self.solve(tripped_lines, use_cache=False)
        return (time.perf_counter() - t0) / repeats * 1000
