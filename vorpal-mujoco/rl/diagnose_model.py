"""Load the diagnosis network saved by diagnose.py; returns f(obs (T, 560)) -> probs (T, 12, NK).
obs is the policy's input at every step of one episode; with ctx > 1 the model also sees the
1 s windows from 1, 2, ... s earlier (stacked here, as in diagnose.py)."""
import pickle

import flax.linen as nn
import jax
import jax.numpy as jp
import numpy as np

import fault_path_env as E


def load(path):
    z = pickle.load(open(path, "rb"))
    mean, std = z["mean"], z["std"]
    lags = np.arange(z.get("ctx", 1)) * E.HIST

    class Diag(nn.Module):                     # must match diagnose.py
        @nn.compact
        def __call__(self, x, train=False):
            x = (x - mean) / std
            for h in (512, 512, 256):
                x = nn.gelu(nn.Dense(h)(x))
                x = nn.Dropout(0.1, deterministic=not train)(x)
            return nn.Dense(12 * E.NK)(x).reshape(x.shape[:-1] + (12, E.NK))

    model = Diag()
    f = jax.jit(lambda x: jax.nn.softmax(model.apply(z["params"], x), -1))

    def run(obs):
        obs = np.asarray(obs)
        t = np.arange(len(obs))
        x = np.concatenate([obs[np.maximum(t - l, 0)] for l in lags], -1)
        return np.asarray(f(jp.asarray(x, jp.float32)))
    return run
