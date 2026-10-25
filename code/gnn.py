import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool
from torch_geometric.utils import scatter

from config import CACHE, GRID_CASE, RESULTS_DIR, SEED, STATES, V_NORMAL, WEIGHTS
from grid import GridSimulator
from reporting import print_gnn_training_summary


def sample_contingencies(num_lines, k, n, rng, exclude=()):
    seen, out = set(exclude), []
    while len(out) < n:
        combo = tuple(sorted(rng.sample(range(num_lines), k)))
        if combo not in seen:
            seen.add(combo)
            out.append(list(combo))
    return out


def build_dataset(sim, n2=800, n3=800, seed=SEED):
    rng = random.Random(seed)
    contingencies = [[]] + [[i] for i in range(sim.num_lines)]
    contingencies += sample_contingencies(sim.num_lines, 2, n2, rng)
    contingencies += sample_contingencies(sim.num_lines, 3, n3, rng)
    data = [sim.to_data(c) for c in contingencies]
    sim.save_cache()
    return data


def split(data, seed=SEED, val=0.1, test=0.2):
    data = list(data)
    random.Random(seed).shuffle(data)
    n_test, n_val = int(len(data) * test), int(len(data) * val)
    return data[n_test + n_val:], data[n_test:n_test + n_val], data[:n_test]


def true_label(d):
    return STATES[int(d.label)]


def class_counts(data):
    counts = {}
    for d in data:
        counts[true_label(d)] = counts.get(true_label(d), 0) + 1
    return counts


class PowerGridGNN(torch.nn.Module):
    def __init__(self, node_features=6, edge_features=2, hidden_dim=64):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(node_features))
        self.register_buffer("x_std", torch.ones(node_features))
        self.conv1 = GATConv(node_features, hidden_dim, edge_dim=edge_features)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_features)
        self.conv3 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_features)
        self.v_head = torch.nn.Linear(hidden_dim, 1)
        self.class_head = torch.nn.Sequential(torch.nn.Linear(2 * hidden_dim + 1, hidden_dim), torch.nn.ReLU(),
                                              torch.nn.Linear(hidden_dim, len(STATES)))

    def set_normalization(self, x):
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp_min(1e-6))

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        base_v = x[:, 0]
        h = (x - self.x_mean) / self.x_std
        h = F.relu(self.conv1(h, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr)) + h
        h = F.relu(self.conv3(h, edge_index, edge_attr)) + h
        v_pred = base_v + self.v_head(h).squeeze(-1)
        min_v = scatter(v_pred, batch, dim=0, dim_size=int(batch.max()) + 1, reduce="min").unsqueeze(-1)
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch), (min_v - V_NORMAL) * 100], dim=1)
        return v_pred, self.class_head(pooled)


def load_model(device="cpu", path=WEIGHTS):
    model = PowerGridGNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def loss_fn(v_pred, class_logits, batch, class_weights):
    node_ok = ~torch.isnan(batch.y)
    v_loss = F.huber_loss(v_pred[node_ok], batch.y[node_ok], delta=0.05) if node_ok.any() else 0.0
    return v_loss * 100 + F.cross_entropy(class_logits, batch.label, weight=class_weights)


def train(model, train_data, epochs=300, lr=3e-3, device="cpu"):
    counts = torch.bincount(torch.tensor([int(d.label) for d in train_data]), minlength=len(STATES)).float()
    class_weights = torch.where(counts > 0, counts.sum() / (len(STATES) * counts.clamp_min(1)), 0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loader = DataLoader(train_data, batch_size=32, shuffle=True)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            v_pred, logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss_fn(v_pred, logits, batch, class_weights).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, data, device="cpu"):
    model.eval()
    abs_err, n_nodes, correct, confusion = 0.0, 0, 0, {}
    for d in data:
        d = d.to(device)
        v_pred, logits = model(d.x, d.edge_index, d.edge_attr)
        truth, pred = true_label(d), STATES[int(logits.argmax())]
        correct += truth == pred
        confusion[f"{truth}->{pred}"] = confusion.get(f"{truth}->{pred}", 0) + 1
        if not d.collapse.item():
            abs_err += (v_pred - d.y).abs().sum().item()
            n_nodes += d.y.numel()
    return {"samples": len(data), "voltage_mae_pu": abs_err / max(n_nodes, 1),
            "semantic_accuracy": correct / max(len(data), 1),
            "confusion_truth_to_pred": dict(sorted(confusion.items()))}


class SemanticGNNOracle:
    def __init__(self, trained_model, sim, device="cpu"):
        self.model = trained_model.to(device)
        self.model.eval()
        self.device = device
        self.sim = sim
        self.x = sim.node_features.to(device)
        self.edge_index = sim.edge_index.to(device)
        self.edge_attr = sim.edge_attr.to(device)
        self.edge_line = sim.edge_line.to(device)
        self.cache = {}

    def predict(self, tripped_lines):
        mask = torch.ones(self.edge_index.shape[1], dtype=torch.bool, device=self.device)
        for i in set(tripped_lines):
            mask &= self.edge_line != i
        with torch.no_grad():
            v, logits = self.model(self.x, self.edge_index[:, mask], self.edge_attr[mask])
        return v.min().item(), STATES[int(logits.argmax())]

    def predict_batch(self, trip_lists, return_probs=False):
        b, n, e = len(trip_lists), self.x.shape[0], self.edge_index.shape[1]
        trips = torch.full((b, max(1, max(len(t) for t in trip_lists))), -1, dtype=torch.long)
        for i, t in enumerate(trip_lists):
            if t:
                trips[i, :len(t)] = torch.tensor(list(t))
        trips = trips.to(self.device)
        keep = ~(self.edge_line[None, None, :] == trips[:, :, None]).any(1)
        offsets = (torch.arange(b, device=self.device) * n)[:, None, None]
        ei = (self.edge_index[None] + offsets).permute(1, 0, 2).reshape(2, b * e)
        flat_keep = keep.reshape(-1)
        batch = torch.arange(b, device=self.device).repeat_interleave(n)
        with torch.no_grad():
            v, logits = self.model(self.x.repeat(b, 1), ei[:, flat_keep], self.edge_attr.repeat(b, 1)[flat_keep], batch)
        min_v = torch.zeros(b, device=self.device).scatter_reduce(0, batch, v, reduce="amin", include_self=False)
        states = [STATES[i] for i in logits.argmax(1).tolist()]
        if return_probs:
            return min_v.cpu(), states, torch.softmax(logits, dim=1).cpu()
        return min_v.cpu(), states

    def semantic_state(self, tripped_lines):
        k = frozenset(tripped_lines)
        if k not in self.cache:
            self.cache[k] = self.predict(tripped_lines)[1]
        return self.cache[k]


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sim = GridSimulator(GRID_CASE, cache_path=CACHE)
    dataset = build_dataset(sim)
    train_data, _, test_data = split(dataset)
    model = PowerGridGNN().to(device)
    model.set_normalization(sim.node_features.to(device))
    t0 = time.perf_counter()
    train(model, train_data, device=device)
    seconds = time.perf_counter() - t0
    metrics = evaluate(model, test_data, device)
    metrics.update(class_counts_all=class_counts(dataset), train_seconds=seconds, train_device=str(device))
    with open(os.path.join(RESULTS_DIR, "gnn_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), WEIGHTS)
    print_gnn_training_summary(len(train_data), seconds, device, metrics["semantic_accuracy"],
                               metrics["voltage_mae_pu"])
