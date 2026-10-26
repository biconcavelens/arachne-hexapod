"""
Train the perfect-motor walking policy (Brax PPO on the MJX env).

    wsl -d Ubuntu -- bash /mnt/c/projects/arachne/vorpal-mujoco/rl/wslrun.sh \
        python train.py name=imu steps=25000000 envs=4096 [imu=1]

Writes runs/<name>/params.pkl (full Brax params) and runs/<name>/policy.npz
(normaliser + MLP weights, NumPy-only inference: play_policy.py / the Uno Q).
"""
import functools
import os
import pickle
import sys
import time

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

import common as C
import hexapod_env

# Brax 0.14 calls jax.device_put_replicated, which JAX 0.11 removed. One GPU:
# replicating = adding a leading axis of size 1 (what pmap expects).
if not hasattr(jax, "device_put_replicated"):
    jax.device_put_replicated = lambda x, devices: jax.tree_util.tree_map(
        lambda a: jax.numpy.stack([a] * len(devices)), x)

args = dict(a.split("=", 1) for a in sys.argv[1:])
NAME = args.get("name", "imu")
STEPS = int(float(args.get("steps", 25_000_000)))
ENVS = int(args.get("envs", 4096))
USE_IMU = args.get("imu", "1") == "1"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", NAME)
os.makedirs(out, exist_ok=True)

cfg = hexapod_env.default_config()
overrides = {"use_imu": USE_IMU}
if "yaw" in args:                                   # max |yaw-rate command|, rad/s
    overrides["command_config.a"] = [float(C.CMD_MAX[0]), float(C.CMD_MAX[1]), float(args["yaw"])]
if "sigma_ang" in args:                             # yaw-tracking reward width, (rad/s)^2
    overrides["reward_config.sigma_ang"] = float(args["sigma_ang"])
env = hexapod_env.Hexapod(cfg, config_overrides=overrides)
print(f"run {NAME}: imu={USE_IMU} steps={STEPS:,} envs={ENVS}  obs {env.observation_size}", flush=True)

t0 = time.time()
LOG = os.path.join(out, "log.txt")


def log(s):
    print(s, flush=True)
    open(LOG, "a").write(s + "\n")


def progress(step, m):
    # eval/episode_* are sums over the 400-step episode
    log(f"[{time.time()-t0:6.0f}s] step {step:>11,}  reward {m['eval/episode_reward']:7.2f}"
        f"  len {m['eval/avg_episode_length']:4.0f}"
        f"  mean speed err {m.get('eval/episode_speed_error', float('nan'))/cfg.episode_length*1000:5.0f} mm/s"
        f"  lin {m.get('eval/episode_reward/tracking_lin_vel', float('nan'))/cfg.episode_length:5.3f}"
        f"  air {m.get('eval/episode_reward/feet_air_time', float('nan')):6.2f}")


networks = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(128, 128, 128),
    value_hidden_layer_sizes=(256, 256, 256),
    policy_obs_key="state",
    value_obs_key="privileged_state",
)
make_policy, params, _ = ppo.train(
    environment=env,
    num_timesteps=STEPS, num_evals=10, reward_scaling=1.0, episode_length=cfg.episode_length,
    normalize_observations=True, action_repeat=1, unroll_length=20, num_minibatches=32,
    num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2,
    num_envs=ENVS, batch_size=256, max_grad_norm=1.0, network_factory=networks,
    seed=int(args.get("seed", 0)), progress_fn=progress, wrap_env_fn=wrapper.wrap_for_brax_training,
)
pickle.dump(jax.tree_util.tree_map(np.asarray, params), open(os.path.join(out, "params.pkl"), "wb"))

# --- export for NumPy inference: params = (normaliser, policy, value)
norm, policy = params[0], params[1]
flat = {"obs_mean": np.asarray(norm.mean["state"]), "obs_std": np.asarray(norm.std["state"])}
layers = policy["params"]
for i, name in enumerate(sorted(layers, key=lambda s: int(s.split("_")[-1]))):
    flat[f"W{i}"], flat[f"b{i}"] = np.asarray(layers[name]["kernel"]), np.asarray(layers[name]["bias"])
np.savez(os.path.join(out, "policy.npz"), **flat)
print("saved", out, "layers:", [(k, v.shape) for k, v in flat.items() if k.startswith("W")], flush=True)
