"""
Spatial–Temporal Graph–Evidential Deep Learning for Uncertainty‑Aware RUL Prognosis
from Acoustic Emission (AE) data on reinforced concrete beams.

Starter implementation (PyTorch) you can adapt to your dataset.

How to use quickly
------------------
1) Update DATA_PATHS below to your CSV files.
   Example (Windows):
   TRAIN = r"F:\\concrete data\\test 3\\per_file+features_800.csv"
   TEST  = r"F:\\concrete data\\test 4\\ae_features_800\\per_file+features_800.csv"

2) Ensure your CSVs have at least these columns (rename in COLS if different):
   - time_step: integer index (monotonic per sensor)
   - sensor_id: integer or string ID
   - features: e.g., ['ae_energy','counts','rise_time','amplitude','duration','mwut_si']
   - target_da: optional; if not present we compute a proxy DA via cumulative sum of mwut_si

3) Provide sensor coordinates for graph construction in SENSOR_COORDS.
   If unknown, set positions along the beam length (e.g., equally spaced) or use a ring graph.

4) Run: the script will
   - load + standardize features,
   - build a spatial adjacency (RBF on coordinates),
   - create spatio‑temporal windows,
   - train a GCN→GRU evidential regressor to predict next‑step DA,
   - evaluate RMSE/MAPE + uncertainty metrics (PICP, calibration error),
   - save model + metrics.

Notes
-----
• Evidential Regression head follows Normal‑Inverse‑Gamma (NIG) parameterization.
• For true RUL, define a failure threshold on DA (e.g., 0.9) and roll forward until crossing.
• This starter keeps dependencies minimal (pure PyTorch, no torch_geometric).
"""

from __future__ import annotations
import os
import math
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Config
# -------------------------
class CFG:
    # Replace with your file paths
    TRAIN_CSV = r"F:\concrete data\test 3\per_file_features_800.csv"
    TEST_CSV  = r"F:\concrete data\test 4\ae_features_800\per_file_features_800.csv"

    # Column names in your CSV
    COLS = {
        'time': 'time_step',
        'sensor': 'sensor_id',
        # features to use (put what's available); mwut_si is optional but recommended
        'features': ['ae_energy','counts','rise_time','amplitude','duration','mwut_si'],
        # optional target damage accumulation column; if missing we derive from mwut_si
        'target': 'target_da',
    }

    # Sensor coordinates (meters) along beam; update to real coordinates if available
    # Example for 8 sensors at 0.2 m spacing starting at 0.4 m
    SENSOR_COORDS: Dict[str, Tuple[float, float]] = {str(i): (0.4 + 0.2*i, 0.0) for i in range(1, 9)}

    # Graph kernel
    RBF_SIGMA = 0.25  # meters; tune based on spacing
    A_THRESH = 1e-3   # prune tiny edges

    # Windowing
    WINDOW = 50  # timesteps input
    HORIZON = 1  # predict t+1 DA

    # Training
    EPOCHS = 40
    BATCH_SIZE = 64
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    SEED = 1337

    # Evidential loss weights
    LAMBDA = 1.0   # evidence regularizer

    # Failure threshold for RUL from DA
    DA_FAIL = 0.9

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)

# -------------------------
# Data loading & preprocessing
# -------------------------

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    return df


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    c = CFG.COLS
    assert c['time'] in df.columns, f"Missing time col {c['time']}"
    assert c['sensor'] in df.columns, f"Missing sensor col {c['sensor']}"
    for f in c['features']:
        if f not in df.columns:
            raise ValueError(f"Feature '{f}' not found; edit CFG.COLS['features'] or your CSV headers")

    # Cast types
    df[c['sensor']] = df[c['sensor']].astype(str)
    df[c['time']] = df[c['time']].astype(int)

    # If target missing, derive DA from mwut_si if present
    tgt = c['target']
    if tgt not in df.columns:
        if 'mwut_si' in c['features'] and 'mwut_si' in df.columns:
            df = df.sort_values([c['sensor'], c['time']])
            df['target_da'] = df.groupby(c['sensor'])['mwut_si'].cumsum()
            # rescale DA per sensor to [0,1]
            df['target_da'] = df.groupby(c['sensor'])['target_da'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
            CFG.COLS['target'] = 'target_da'
        else:
            raise ValueError("No target_da and no mwut_si to derive it. Provide a target column or include mwut_si.")

    # Standardize features per test (robust scaler)
    feats = c['features']
    for f in feats:
        v = df[f].astype(float)
        med = v.median(); iqr = (v.quantile(0.75) - v.quantile(0.25))
        iqr = iqr if iqr > 0 else v.std() + 1e-6
        df[f] = (v - med) / (iqr + 1e-6)
    return df


def pivot_by_time(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """Return tensors of shape [T, N, F] and DA target [T, N]."""
    c = CFG.COLS
    feats = c['features']
    sensors = sorted(df[c['sensor']].unique().tolist())
    times = sorted(df[c['time']].unique().tolist())
    N = len(sensors); Fdim = len(feats); T = len(times)
    X = np.zeros((T, N, Fdim), dtype=np.float32)
    Y = np.zeros((T, N), dtype=np.float32)

    sensor_index = {s: i for i, s in enumerate(sensors)}
    time_index = {t: i for i, t in enumerate(times)}

    for _, row in df.iterrows():
        ti = time_index[row[c['time']]]
        si = sensor_index[row[c['sensor']]]
        X[ti, si, :] = row[feats].values.astype(np.float32)
        Y[ti, si] = float(row[c['target']])

    return X, Y, sensors, times


# -------------------------
# Graph construction (RBF adjacency)
# -------------------------

def build_adj(sensors: List[str]) -> torch.Tensor:
    coords = CFG.SENSOR_COORDS
    pts = []
    for s in sensors:
        if s not in coords:
            # fallback: place missing sensors on a line with equal spacing
            idx = int(s) if s.isdigit() else len(pts)
            pts.append((0.2*idx, 0.0))
        else:
            pts.append(coords[s])
    P = np.array(pts, dtype=np.float32)  # [N,2]
    N = P.shape[0]
    dists = np.linalg.norm(P[None, :, :] - P[:, None, :], axis=-1)  # [N,N]
    A = np.exp(- (dists**2) / (2*CFG.RBF_SIGMA**2))
    np.fill_diagonal(A, 1.0)
    A[A < CFG.A_THRESH] = 0.0
    # Row-normalize
    A = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    return torch.tensor(A, dtype=torch.float32)


# -------------------------
# Dataset
# -------------------------
class STGraphDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, window: int):
        self.X = X  # [T,N,F]
        self.Y = Y  # [T,N]
        self.W = window

    def __len__(self):
        return max(0, self.X.shape[0] - self.W - CFG.HORIZON + 1)

    def __getitem__(self, idx):
        x_win = self.X[idx: idx + self.W]          # [W,N,F]
        y_next = self.Y[idx + self.W]              # [N]
        return torch.tensor(x_win), torch.tensor(y_next)


# -------------------------
# Model: simple GCN + GRU + Evidential head
# -------------------------
class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, hid_dim)
        self.lin_nei  = nn.Linear(in_dim, hid_dim)
        self.act = nn.ELU()

    def forward(self, x, A):
        # x: [B,N,F]
        x_self = self.lin_self(x)
        x_nei  = torch.matmul(A, x)  # [B,N,F] with broadcasting if A [N,N]
        x_nei  = self.lin_nei(x_nei)
        return self.act(x_self + x_nei)

class STGEvidential(nn.Module):
    def __init__(self, in_dim, gcn_hid=64, gru_hid=128):
        super().__init__()
        self.gcn1 = SimpleGCN(in_dim, gcn_hid)
        self.gcn2 = SimpleGCN(gcn_hid, gcn_hid)
        self.gru = nn.GRU(input_size=gcn_hid, hidden_size=gru_hid, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hid, 128), nn.ELU(),
            nn.Linear(128, 4)  # outputs: mu, logv, logalpha, logbeta
        )

    def forward(self, x_seq, A):
        # x_seq: [B,W,N,F]
        B, W, N, Fdim = x_seq.shape
        x_seq = x_seq.reshape(B*W, N, Fdim)
        A = A.to(x_seq.device)
        h = self.gcn1(x_seq, A)
        h = self.gcn2(h, A)               # [B*W,N,H]
        h = h.mean(dim=1)                 # global mean over sensors → [B*W,H]
        h = h.reshape(B, W, -1)           # [B,W,H]
        out, _ = self.gru(h)              # [B,W,gru_hid]
        z = out[:, -1, :]                 # last step
        p = self.head(z)
        mu, logv, logalpha, logbeta = torch.chunk(p, 4, dim=-1)
        v = F.softplus(logv) + 1e-3
        alpha = F.softplus(logalpha) + 1.0 + 1e-3
        beta  = F.softplus(logbeta) + 1e-3
        return mu.squeeze(-1), v.squeeze(-1), alpha.squeeze(-1), beta.squeeze(-1)


# -------------------------
# Evidential Regression (Normal‑Inverse‑Gamma)
# -------------------------

def nig_nll(y, mu, v, alpha, beta):
    # y, mu, v, alpha, beta: [B]
    two_beta_v = 2*beta*(1+v)
    nll = 0.5*torch.log(math.pi/v) - alpha*torch.log(two_beta_v) + (alpha+0.5)*torch.log(v*(y-mu)**2 + two_beta_v) + torch.lgamma(alpha) - torch.lgamma(alpha+0.5)
    return nll

def evidence_regularizer(y, mu, v, alpha, beta):
    # promote higher evidence when error small
    err = torch.abs(y - mu)
    return err * (2*v + alpha)


def evidential_loss(y, mu, v, alpha, beta, lam=1.0):
    return nig_nll(y, mu, v, alpha, beta).mean() + lam * evidence_regularizer(y, mu, v, alpha, beta).mean()


# -------------------------
# Train / Eval
# -------------------------

def train_one_epoch(model, dl, A, opt):
    model.train(); total = 0.0
    for x, yN in dl:
        # Reduce across sensors to a global target by averaging DA across sensors
        y = yN.mean(dim=1)  # [B]
        x = x.to(CFG.DEVICE); y = y.to(CFG.DEVICE)
        mu, v, alpha, beta = model(x, A)
        loss = evidential_loss(y, mu, v, alpha, beta, CFG.LAMBDA)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * x.size(0)
    return total / len(dl.dataset)


def predict(model, dl, A):
    model.eval()
    pred_mu, pred_var, y_true = [], [], []
    with torch.no_grad():
        for x, yN in dl:
            y = yN.mean(dim=1)
            x = x.to(CFG.DEVICE)
            mu, v, alpha, beta = model(x, A)
            # predictive variance of Student‑t (approx): beta/(v*(alpha-1))
            var = beta / (v * (alpha - 1 + 1e-6))
            pred_mu.append(mu.cpu()); pred_var.append(var.cpu()); y_true.append(y.cpu())
    return torch.cat(pred_mu), torch.cat(pred_var), torch.cat(y_true)


def rmse(a,b):
    return float(torch.sqrt(torch.mean((a-b)**2)).item())

def mape(a,b):
    return float((torch.mean(torch.abs((a-b) / (a.abs()+1e-6))).item()))

def picp(y, mu, var, q=0.95):
    # assuming normal approx
    std = torch.sqrt(torch.clamp(var, 1e-8))
    z = torch.tensor(1.959964, device=mu.device) if q==0.95 else torch.tensor(1.644854, device=mu.device)
    lo = mu - z*std; hi = mu + z*std
    inside = ((y >= lo) & (y <= hi)).float().mean().item()
    return inside


def ece_gaussian(y, mu, var, bins=10):
    # simple expected calibration error using normal CDF
    from math import erf, sqrt
    y = y.numpy(); mu = mu.numpy(); std = np.sqrt(np.clip(var.numpy(), 1e-8, None))
    cdf = 0.5*(1 + (y-mu)/(std*np.sqrt(2)))  # rough proxy; not exact CDF
    # binning
    edges = np.linspace(0,1,bins+1)
    ece = 0.0
    for i in range(bins):
        m = (cdf>=edges[i]) & (cdf<edges[i+1])
        if m.any():
            conf = ((edges[i]+edges[i+1])/2.0)
            acc = np.mean((cdf[m] > 0.5).astype(float))
            ece += np.abs(acc - conf) * (m.mean())
    return float(ece)


# -------------------------
# Main
# -------------------------

def main():
    print("Device:", CFG.DEVICE)
    tr = ensure_schema(load_csv(CFG.TRAIN_CSV))
    te = ensure_schema(load_csv(CFG.TEST_CSV))

    Xtr, Ytr, sensors_tr, times_tr = pivot_by_time(tr)
    Xte, Yte, sensors_te, times_te = pivot_by_time(te)

    # Align sensors between train/test (use intersection)
    sensors = sorted(list(set(sensors_tr) & set(sensors_te)))
    if len(sensors) == 0:
        raise RuntimeError("No overlapping sensors between train/test; please harmonize sensor_id labels")

    # Filter to common sensors
    def filter_sensors(X, Y, sensors_all, sensors_keep):
        idx = [sensors_all.index(s) for s in sensors_keep]
        return X[:, idx, :], Y[:, idx]

    Xtr, Ytr = filter_sensors(Xtr, Ytr, sensors_tr, sensors)
    Xte, Yte = filter_sensors(Xte, Yte, sensors_te, sensors)

    A = build_adj(sensors).to(CFG.DEVICE)

    dtr = STGraphDataset(Xtr, Ytr, CFG.WINDOW)
    dte = STGraphDataset(Xte, Yte, CFG.WINDOW)

    ltr = DataLoader(dtr, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)
    lte = DataLoader(dte, batch_size=CFG.BATCH_SIZE, shuffle=False)

    model = STGEvidential(in_dim=len(CFG.COLS['features'])).to(CFG.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)

    best = {"rmse": 1e9}
    for epoch in range(1, CFG.EPOCHS+1):
        tr_loss = train_one_epoch(model, ltr, A, opt)
        mu, var, y = predict(model, lte, A)
        r = rmse(mu, y); m = mape(mu, y); p = picp(y, mu, var, q=0.95)
        if r < best['rmse']:
            best.update({"epoch": epoch, "rmse": r, "mape": m, "picp": p})
            torch.save({"model": model.state_dict(), "cfg": CFG.__dict__}, "stge_rul_best.pt")
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | RMSE={r:.4f} | MAPE={m:.4f} | PICP@95={p:.3f}")

    print("Best:", best)
    with open("stge_rul_metrics.json","w") as f:
        json.dump(best, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
