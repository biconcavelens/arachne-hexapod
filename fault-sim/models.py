"""
Stage B: the healthy-motion model and the diagnosis network (PyTorch).

Per control step the network sees D_IN numbers:
  measured: ego twist + flow grid (38), IMU (9), commands (12), gait phase (2)
  residual: measured ego + IMU minus what the healthy model predicts from the commands (47)
One window = one gait cycle (24 steps). Heads:
  any    fault present in the window
  types  per servo: none / loss / sag / deadband          (handles one or two faulty servos)
  size   per servo: loss drop rate, sag span / 5 deg, dead band / 20 deg  (normalised)
  aux    executed joint angles per step (simulation labels only)
"""
import numpy as np
import torch
import torch.nn as nn

import config as C

D_STEP = C.FEAT_EGO + C.FEAT_IMU + C.FEAT_CMD + C.FEAT_PHASE    # 61
D_PRED = C.FEAT_EGO + C.FEAT_IMU                                # 47
HIST = 6                                                         # command steps the healthy model sees
D_HEALTHY_IN = HIST * 12 + C.FEAT_PHASE
D_IN = D_STEP + D_PRED                                           # 108
IMU_COLS = np.r_[C.FEAT_EGO:D_PRED, D_STEP + C.FEAT_EGO:D_STEP + D_PRED]   # measured IMU and its residual
N_TYPES = len(C.FAULT_TYPES)
SIZE_SCALE = {"none": 1.0, "loss": 1.0, "sag": 5.0, "deadband": 20.0}


def step_inputs(feat, imu, cmd, phase):
    return np.concatenate([feat, imu, cmd, phase], axis=1).astype(np.float32)


def healthy_inputs(cmd, phase):
    """(n, 12) commands -> (n, HIST*12 + 2): this step and the previous HIST-1, padded at the start."""
    c = np.concatenate([np.repeat(cmd[:1], HIST - 1, 0), cmd])
    hist = np.stack([c[i:i + len(cmd)] for i in range(HIST)], axis=1).reshape(len(cmd), -1)
    return np.concatenate([hist, phase], axis=1).astype(np.float32)


class HealthyModel(nn.Module):
    """Commands -> the ego and IMU features a healthy robot would produce."""

    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_HEALTHY_IN, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, D_PRED))

    def forward(self, x):
        return self.net(x)


class Diagnoser(nn.Module):
    def __init__(self, arch="transformer", width=128, layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.arch = arch
        self.inp = nn.Linear(D_IN, width)
        if arch == "transformer":
            self.pos = nn.Parameter(torch.zeros(1, C.WINDOW, width))
            layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout, batch_first=True, activation="gelu")
            self.body = nn.TransformerEncoder(layer, layers)
        else:
            self.body = nn.GRU(width, width, num_layers=2, batch_first=True, dropout=dropout)
        self.any = nn.Linear(width, 1)
        self.types = nn.Linear(width, 12 * N_TYPES)
        self.size = nn.Linear(width, 12)
        self.aux = nn.Linear(width, 12)

    def forward(self, x):
        h = self.inp(x)
        if self.arch == "transformer":
            h = self.body(h + self.pos[:, :h.shape[1]])
            pooled = h.mean(1)
        else:
            h, _ = self.body(h)
            pooled = h[:, -1]
        return {"any": self.any(pooled).squeeze(-1), "types": self.types(pooled).view(-1, 12, N_TYPES),
                "size": self.size(pooled), "aux": self.aux(h)}


def loss_fn(out, batch, type_weights, w_size=1.0, w_aux=0.1):
    l_any = nn.functional.binary_cross_entropy_with_logits(out["any"], batch["any"])
    l_type = nn.functional.cross_entropy(out["types"].reshape(-1, N_TYPES), batch["types"].reshape(-1), weight=type_weights)
    faulty = batch["types"] > 0
    l_size = nn.functional.huber_loss(out["size"][faulty], batch["size"][faulty]) if faulty.any() else out["size"].sum() * 0
    l_aux = nn.functional.mse_loss(out["aux"], batch["q"])
    total = l_any + l_type + w_size * l_size + w_aux * l_aux
    return total, {"any": l_any.item(), "type": l_type.item(), "size": l_size.item(), "aux": l_aux.item()}


def save(path, healthy, diag, norm, cfg):
    torch.save({"healthy": healthy.state_dict(), "diag": diag.state_dict(), "norm": norm, "cfg": cfg}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    healthy, diag = HealthyModel(), Diagnoser(**ck["cfg"]["diag"])
    healthy.load_state_dict(ck["healthy"])
    diag.load_state_dict(ck["diag"])
    return healthy.eval().to(device), diag.eval().to(device), ck["norm"], ck["cfg"]


def window_inputs(step, healthy_in, healthy, norm, device="cpu"):
    """Measured step inputs (n, D_STEP) + healthy-model inputs -> normalised network inputs (n, D_IN)."""
    with torch.no_grad():
        hx = torch.from_numpy((healthy_in - norm["h_mu"]) / norm["h_sd"]).to(device)
        pred = healthy(hx).cpu().numpy() * norm["p_sd"] + norm["p_mu"]
    resid = step[:, :D_PRED] - pred
    x = np.concatenate([(step - norm["s_mu"]) / norm["s_sd"], (resid - norm["r_mu"]) / norm["r_sd"]], axis=1).astype(np.float32)
    if norm.get("no_imu"):                # no IMU: its inputs sit at the training mean (0 once normalised)
        x[:, IMU_COLS] = 0
    return x
