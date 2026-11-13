"""
Stage A: camera frames -> ego-motion. No learning and no knowledge of faults.

1. Optical flow between consecutive frames (OpenCV DIS on the CPU; RAFT optional).
2. Planar ego-motion: below the horizon the camera sees the floor, whose depth per
   pixel is known from the camera height and tilt. The flow there is linear in the
   camera's 6-D velocity (Longuet-Higgins and Prazdny motion field), so a robust
   least-squares fit gives a metric twist without a learned depth model.
3. A coarse summary: mean flow on a GRID x GRID grid.

Output per frame: body-frame twist (vx, vy, vz, wx, wy, wz) and 2*GRID*GRID flow values.
Camera convention for the maths: OpenCV (x right, y down, z forward).
"""
import math

import numpy as np

import config as C


def intrinsics(w=C.IMG_W, h=C.IMG_H, fovy_deg=C.FOVY_DEG):
    fy = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
    return np.array([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1.0]])


def camera_axes_in_body(pose):
    """Columns: camera x (right), y (down), z (forward) in the base-body frame."""
    fwd, up = np.array(pose["forward"]), np.array(pose["up"])
    right = np.cross(fwd, up)
    return np.c_[right, -up, fwd]


class PlaneEgomotion:
    """Fits the camera twist to the flow on the floor."""

    def __init__(self, K, R_bc, cam_pos_body, cam_height, step=4, max_range=3.0):
        self.K, self.R_bc, self.r_c = K, R_bc, np.asarray(cam_pos_body, float)
        h, w = int(K[1, 2] * 2), int(K[0, 2] * 2)
        v, u = np.mgrid[step // 2:h:step, step // 2:w:step]
        x = (u - K[0, 2]) / K[0, 0]
        y = (v - K[1, 2]) / K[1, 1]
        down_cam = R_bc.T @ np.array([0, 0, -1.0])       # world down, in camera axes (body assumed level)
        ndotr = down_cam[0] * x + down_cam[1] * y + down_cam[2]
        Z = np.where(ndotr > 1e-3, cam_height / np.maximum(ndotr, 1e-3), np.inf)
        keep = np.isfinite(Z) & (Z < max_range)
        self.u, self.v = u[keep], v[keep]
        self.x, self.y, self.Z = x[keep], y[keep], Z[keep]
        x, y, iz = self.x, self.y, 1 / self.Z
        zeros = np.zeros_like(x)
        # rows: d(x)/dt and d(y)/dt as linear functions of (Vx, Vy, Vz, wx, wy, wz)
        self.A = np.r_[np.c_[-iz, zeros, x * iz, x * y, -(1 + x * x), y],
                       np.c_[zeros, -iz, y * iz, 1 + y * y, -x * y, -x]]

    def solve(self, flow, dt, iters=3, huber=1.5):
        """flow: (H, W, 2) pixels per frame -> body-frame twist (v at the base origin, w)."""
        du = flow[self.v, self.u, 0] / self.K[0, 0] / dt
        dv = flow[self.v, self.u, 1] / self.K[1, 1] / dt
        b = np.r_[du, dv]
        w = np.ones_like(b)
        theta = np.zeros(6)
        for _ in range(iters):                                  # iteratively reweighted least squares
            sw = np.sqrt(w)
            theta = np.linalg.lstsq(self.A * sw[:, None], b * sw, rcond=None)[0]
            r = np.abs(self.A @ theta - b)
            s = 1.4826 * np.median(r) + 1e-9
            w = np.where(r <= huber * s, 1.0, huber * s / np.maximum(r, 1e-12))
        V_c, W_c = theta[:3], theta[3:]
        w_b = self.R_bc @ W_c
        v_b = self.R_bc @ V_c - np.cross(w_b, self.r_c)         # camera velocity -> base-origin velocity
        return np.r_[v_b, w_b]

    def flow_from_twist(self, twist_body, dt):
        """Predicted floor flow (pixels per frame) for a body twist. Used by the tests and the no-render mode."""
        v_b, w_b = twist_body[:3], twist_body[3:]
        V_c = self.R_bc.T @ (v_b + np.cross(w_b, self.r_c))
        W_c = self.R_bc.T @ w_b
        d = self.A @ np.r_[V_c, W_c] * dt
        n = len(self.x)
        return d[:n] * self.K[0, 0], d[n:] * self.K[1, 1]


def grid_summary(flow, grid=C.GRID):
    """Mean (u, v) flow per cell, normalised by the image width."""
    h, w = flow.shape[:2]
    out = []
    for i in range(grid):
        for j in range(grid):
            cell = flow[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid]
            out += [cell[..., 0].mean() / w, cell[..., 1].mean() / w]
    return np.array(out, dtype=np.float32)


def grid_from_twist(pe, twist_body, dt, w=C.IMG_W, h=C.IMG_H, grid=C.GRID):
    """The same grid summary, predicted from a twist (floor cells only; sky cells stay 0)."""
    fu, fv = pe.flow_from_twist(twist_body, dt)
    out = np.zeros(2 * grid * grid, dtype=np.float32)
    ci = np.minimum(pe.v * grid // h, grid - 1) * grid + np.minimum(pe.u * grid // w, grid - 1)
    for c in range(grid * grid):
        m = ci == c
        if m.any():
            out[2 * c], out[2 * c + 1] = fu[m].mean() / w, fv[m].mean() / w
    return out


class FlowEstimator:
    """Optical flow, 'dis' (OpenCV, CPU) or 'raft' (torchvision RAFT-small, GPU if available)."""

    def __init__(self, method="dis"):
        self.method = method
        if method == "dis":
            import cv2
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        else:
            import torch
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
            self.torch = torch
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.net = raft_small(weights=Raft_Small_Weights.DEFAULT).eval().to(self.dev)

    def __call__(self, prev, nxt):
        if self.method == "dis":
            return self.dis.calc(prev, nxt, None)
        t = self.torch
        def prep(img):                                   # gray uint8 -> 3x(H8)x(W8) in [-1, 1]
            x = t.from_numpy(img).float().div(127.5).sub(1).expand(3, -1, -1)[None]
            h8, w8 = (img.shape[0] + 7) // 8 * 8, (img.shape[1] + 7) // 8 * 8
            return t.nn.functional.pad(x, (0, w8 - img.shape[1], 0, h8 - img.shape[0]), mode="replicate").to(self.dev)
        with t.no_grad():
            f = self.net(prep(prev), prep(nxt))[-1][0, :, :prev.shape[0], :prev.shape[1]]
        return f.permute(1, 2, 0).cpu().numpy()


class StageA:
    """Frame stream -> per-step features (twist 6 + flow grid 32)."""

    def __init__(self, pose, cam_height, flow_method="dis"):
        self.K = intrinsics()
        self.pe = PlaneEgomotion(self.K, camera_axes_in_body(pose), pose["pos_m"], cam_height)
        self.flow = FlowEstimator(flow_method)
        self.prev = None

    def update(self, gray):
        if self.prev is None:
            self.prev = gray
            return np.zeros(C.FEAT_EGO, dtype=np.float32)
        f = self.flow(self.prev, gray)
        self.prev = gray
        return np.r_[self.pe.solve(f, C.CTRL_DT), grid_summary(f)].astype(np.float32)
