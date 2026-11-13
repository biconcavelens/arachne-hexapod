"""
Train Stage B on episodes from generate.py: the healthy-motion model first, then the
diagnosis network. Reports metrics on held-out episodes and saves <out>/model.pt.

    python train.py --data data/gt --out runs/gt_transformer
    python train.py --data data/gt --out runs/gt_gru --arch gru
    python train.py --data data/gt data/flow --out runs/mixed --baseline   # + the pilot's logistic regression
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config as C
import models as M

TYPE_NAMES = list(C.FAULT_TYPES)


def load_episodes(dirs):
    eps = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "ep_*.npz"))):
            z = np.load(f)
            ep = {k: z[k] for k in ("feat", "imu", "cmd", "phase", "q", "types", "sizes")}
            ep["step"] = M.step_inputs(ep["feat"], ep["imu"], ep["cmd"], ep["phase"])
            ep["hin"] = M.healthy_inputs(ep["cmd"], ep["phase"])
            scale = np.array([[M.SIZE_SCALE[TYPE_NAMES[t]] for t in row] for row in ep["types"]], np.float32)
            ep["size_n"] = ep["sizes"] / scale
            eps.append(ep)
    return eps


def window_index(eps):
    """(episode, start) for every window after warm-up whose labels do not change inside it."""
    w0, idx = int(C.WARMUP_S / C.CTRL_DT), []
    for e, ep in enumerate(eps):
        ty = ep["types"]
        for s in range(w0, len(ty) - C.WINDOW + 1, C.STRIDE):
            if (ty[s:s + C.WINDOW] == ty[s + C.WINDOW - 1]).all():
                idx.append((e, s))
    return np.array(idx, dtype=np.int64)


def fit_healthy(eps, device, epochs, log):
    w0 = int(C.WARMUP_S / C.CTRL_DT)
    X = np.concatenate([ep["hin"][w0:][(ep["types"][w0:] == 0).all(1)] for ep in eps])
    Y = np.concatenate([ep["step"][w0:, :M.D_PRED][(ep["types"][w0:] == 0).all(1)] for ep in eps])
    norm = {"h_mu": X.mean(0), "h_sd": X.std(0) + 1e-6, "p_mu": Y.mean(0), "p_sd": Y.std(0) + 1e-6}
    net = M.HealthyModel().to(device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt = torch.from_numpy((X - norm["h_mu"]) / norm["h_sd"]).float().to(device)
    Yt = torch.from_numpy((Y - norm["p_mu"]) / norm["p_sd"]).float().to(device)
    for ep_i in range(epochs):
        perm = torch.randperm(len(Xt), device=device)
        tot = 0.0
        for i in range(0, len(Xt), 2048):
            b = perm[i:i + 2048]
            loss = torch.nn.functional.mse_loss(net(Xt[b]), Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep_i % 5 == 0 or ep_i == epochs - 1:
            log(f"  healthy model epoch {ep_i}: mse {tot / len(Xt):.4f} (normalised, on {len(Xt)} healthy steps)")
    return net.eval(), norm


class Windows(torch.utils.data.Dataset):
    def __init__(self, eps, idx, X):
        self.eps, self.idx, self.X = eps, idx, X

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        e, s = self.idx[i]
        ep, end = self.eps[e], s + C.WINDOW - 1
        ty = ep["types"][end]
        return {"x": torch.from_numpy(self.X[e][s:s + C.WINDOW]), "any": torch.tensor(float((ty > 0).any())),
                "types": torch.from_numpy(ty), "size": torch.from_numpy(ep["size_n"][end]),
                "q": torch.from_numpy(ep["q"][s:s + C.WINDOW])}


def predict(diag, loader, device):
    outs = {"any": [], "types": [], "size": []}
    labels = {"any": [], "types": [], "size": []}
    with torch.no_grad():
        for b in loader:
            o = diag(b["x"].to(device))
            outs["any"].append(torch.sigmoid(o["any"]).cpu().numpy())
            outs["types"].append(torch.softmax(o["types"], -1).cpu().numpy())
            outs["size"].append(o["size"].cpu().numpy())
            for k in labels:
                labels[k].append(b[k].numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}, {k: np.concatenate(v) for k, v in labels.items()}


def metrics(out, lab):
    p_type, y_type = out["types"], lab["types"]
    p_fault = 1 - p_type[..., 0]
    true_set = y_type > 0
    n_true = true_set.sum(1)
    res = {"windows": int(len(y_type)), "faulty_windows": int((n_true > 0).sum())}
    if 0 < lab["any"].sum() < len(lab["any"]):
        res["detect_auroc"] = float(roc_auc_score(lab["any"], out["any"]))
    single = n_true == 1
    if single.any():
        res["which_servo_acc_single_fault"] = float((p_fault[single].argmax(1) == true_set[single].argmax(1)).mean())
    faulty = n_true > 0
    if faulty.any():
        res["exact_set_acc"] = float(((p_fault[faulty] > 0.5) == true_set[faulty]).all(1).mean())
        pred_t = p_type.argmax(-1)
        res["type_acc_on_faulty_servos"] = float((pred_t[true_set] == y_type[true_set]).mean())
        conf = np.zeros((len(TYPE_NAMES), len(TYPE_NAMES)), int)
        np.add.at(conf, (y_type.ravel(), pred_t.ravel()), 1)
        res["type_confusion_rows_true_cols_pred"] = {"labels": TYPE_NAMES, "counts": conf.tolist()}
        ok = true_set & (pred_t == y_type)
        for ti, name in enumerate(TYPE_NAMES[1:], 1):
            m = ok & (y_type == ti)
            if m.any():
                res[f"size_mae_{name}"] = float(np.abs(out["size"][m] - lab["size"][m]).mean() * M.SIZE_SCALE[name])
    return res


def baseline(eps, tr, va, X):
    """The pilot's approach: logistic regression on the flattened residual window."""
    def flat(idx, cap=30000):
        idx = idx[np.random.default_rng(0).permutation(len(idx))[:cap]]
        F = np.stack([X[e][s:s + C.WINDOW, M.D_STEP:].ravel() for e, s in idx])
        ty = np.stack([eps[e]["types"][s + C.WINDOW - 1] for e, s in idx])
        return F, ty
    Ftr, ttr = flat(tr)
    Fva, tva = flat(va)
    res = {}
    det = LogisticRegression(max_iter=2000, C=0.05).fit(Ftr, (ttr > 0).any(1))
    res["baseline_detect_auroc"] = float(roc_auc_score((tva > 0).any(1), det.predict_proba(Fva)[:, 1]))
    s_tr, s_va = (ttr > 0).sum(1) == 1, (tva > 0).sum(1) == 1
    if not (s_tr.any() and s_va.any()):
        return res
    loc =LogisticRegression(max_iter=2000, C=0.05).fit(Ftr[s_tr], (ttr[s_tr] > 0).argmax(1))
    res["baseline_which_servo_acc_single_fault"] = float((loc.predict(Fva[s_va]) == (tva[s_va] > 0).argmax(1)).mean())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=["transformer", "gru"], default="transformer")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--healthy-epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--no-imu", action="store_true", help="camera-only model: IMU inputs held at their mean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", metavar="CKPT", help="no training: score this model on the validation split of --data "
                                                   "(e.g. a model trained in another environment stage)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logf = open(os.path.join(a.out, "log.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    t0 = time.time()
    eps = load_episodes(a.data)
    perm = np.random.default_rng(a.seed).permutation(len(eps))
    n_va = max(1, int(len(eps) * a.val_frac))
    va_eps, tr_eps = set(perm[:n_va].tolist()), set(perm[n_va:].tolist())
    log(f"{len(eps)} episodes ({len(tr_eps)} train / {len(va_eps)} val), device {device}")
    if a.eval:
        healthy, diag, norm, _ = M.load(a.eval, device)
        X = [M.window_inputs(ep["step"], ep["hin"], healthy, norm, device) if i in va_eps else None for i, ep in enumerate(eps)]
        idx = window_index(eps)
        va = idx[np.isin(idx[:, 0], list(va_eps))]
        res = metrics(*predict(diag, torch.utils.data.DataLoader(Windows(eps, va, X), batch_size=1024), device))
        json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
        log(f"evaluated {a.eval}: " + json.dumps({k: round(v, 3) for k, v in res.items() if isinstance(v, float)}))
        raise SystemExit

    healthy, norm = fit_healthy([eps[i] for i in tr_eps], device, a.healthy_epochs, log)
    with torch.no_grad():
        resid_tr = []
        for i in tr_eps:
            hx = torch.from_numpy((eps[i]["hin"] - norm["h_mu"]) / norm["h_sd"]).float().to(device)
            pred = healthy(hx).cpu().numpy() * norm["p_sd"] + norm["p_mu"]
            resid_tr.append(eps[i]["step"][:, :M.D_PRED] - pred)
    S = np.concatenate([eps[i]["step"] for i in tr_eps])
    Rr = np.concatenate(resid_tr)
    norm.update(s_mu=S.mean(0), s_sd=S.std(0) + 1e-6, r_mu=Rr.mean(0), r_sd=Rr.std(0) + 1e-6)
    norm = {k: v.astype(np.float32) for k, v in norm.items()}
    norm["no_imu"] = a.no_imu                   # saved with the model, so inference does the same
    healthy = healthy.float()
    X = [M.window_inputs(ep["step"], ep["hin"], healthy, norm, device) for ep in eps]

    idx = window_index(eps)
    tr = idx[np.isin(idx[:, 0], list(tr_eps))]
    va = idx[np.isin(idx[:, 0], list(va_eps))]
    counts = np.bincount(np.concatenate([eps[e]["types"][s + C.WINDOW - 1] for e, s in tr]), minlength=M.N_TYPES)
    type_w = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    type_w = (type_w / type_w.mean()).to(device)
    log(f"windows: {len(tr)} train / {len(va)} val; type counts (servo slots, train) {dict(zip(TYPE_NAMES, counts.tolist()))}")

    dl_tr = torch.utils.data.DataLoader(Windows(eps, tr, X), batch_size=a.batch, shuffle=True, num_workers=0)
    dl_va = torch.utils.data.DataLoader(Windows(eps, va, X), batch_size=1024, shuffle=False, num_workers=0)
    cfg = {"diag": {"arch": a.arch}, "args": vars(a)}
    diag = M.Diagnoser(a.arch).to(device)
    opt = torch.optim.AdamW(diag.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best = (np.inf, None)
    for ep_i in range(a.epochs):
        diag.train()
        parts = {}
        for b in dl_tr:
            b = {k: v.to(device) for k, v in b.items()}
            loss, p = M.loss_fn(diag(b["x"]), b, type_w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diag.parameters(), 1.0)
            opt.step()
            for k, v in p.items():
                parts[k] = parts.get(k, 0.0) + v / len(dl_tr)
        sched.step()
        diag.eval()
        with torch.no_grad():
            vl = np.mean([M.loss_fn(diag(b["x"].to(device)), {k: v.to(device) for k, v in b.items()}, type_w)[0].item()
                          for b in dl_va])
        if vl < best[0]:
            best = (vl, {k: v.detach().cpu().clone() for k, v in diag.state_dict().items()})
        log(f"epoch {ep_i}: train {' '.join(f'{k} {v:.3f}' for k, v in parts.items())} | val loss {vl:.3f} ({time.time() - t0:.0f}s)")

    diag.load_state_dict(best[1])
    diag.eval()
    out, lab = predict(diag, dl_va, device)
    res = metrics(out, lab)
    if a.baseline:
        res.update(baseline(eps, tr, va, X))
    if not a.no_imu:                            # the same model with the IMU missing at test time
        Xd = [M.window_inputs(ep["step"], ep["hin"], healthy, dict(norm, no_imu=True), device) if i in va_eps else None
              for i, ep in enumerate(eps)]
        dl_d = torch.utils.data.DataLoader(Windows(eps, va, Xd), batch_size=1024, shuffle=False, num_workers=0)
        res["imu_dropped_at_test"] = metrics(*predict(diag, dl_d, device))
        log("IMU dropped at test: " + json.dumps({k: round(v, 3) for k, v in res["imu_dropped_at_test"].items()
                                                  if isinstance(v, float)}))
    M.save(os.path.join(a.out, "model.pt"), healthy.cpu(), diag.cpu(), norm, cfg)
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    log(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))
    log(f"saved {a.out}/model.pt in {time.time() - t0:.0f}s")
