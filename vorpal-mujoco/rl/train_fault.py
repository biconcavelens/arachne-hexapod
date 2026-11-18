"""
Train the fault-tolerant path-following policy (Brax PPO on fault_path_env).

    bash wslrun.sh python train_fault.py name=fault steps=60000000 envs=8192 [faults=1] [seed=0]

faults=0 trains the same task with healthy servos only (the "no fault training" baseline).
Writes runs/<name>/params.pkl (latest checkpoint, refreshed at every evaluation),
runs/<name>/config.json and runs/<name>/log.txt.
"""
import functools
import json
import os
import pickle
import sys
import time

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

import fault_path_env as E

if not hasattr(jax, "device_put_replicated"):         # Brax 0.14 vs JAX 0.11
    jax.device_put_replicated = lambda x, devices: jax.tree_util.tree_map(
        lambda a: jax.numpy.stack([a] * len(devices)), x)

args = dict(a.split("=", 1) for a in sys.argv[1:])
NAME = args.get("name", "fault")
STEPS = int(float(args.get("steps", 60_000_000)))
ENVS = int(args.get("envs", 8192))
FAULTS = args.get("faults", "1") == "1"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", NAME)
os.makedirs(out, exist_ok=True)

overrides = {} if FAULTS else {"fault_config.p_count": [1.0, 0.0, 0.0, 0.0]}
env = E.FaultPath(config_overrides=overrides)
cfg = env._config
POLICY_LAYERS, VALUE_LAYERS = (512, 256, 128), (512, 256, 256)
json.dump({"faults": FAULTS, "steps": STEPS, "envs": ENVS, "policy_layers": POLICY_LAYERS,
           "value_layers": VALUE_LAYERS, "config": cfg.to_dict()}, open(os.path.join(out, "config.json"), "w"), indent=1)
print(f"run {NAME}: faults={FAULTS} steps={STEPS:,} envs={ENVS} obs {env.observation_size}", flush=True)

t0 = time.time()
LOG = os.path.join(out, "log.txt")


def log(s):
    print(s, flush=True)
    open(LOG, "a").write(s + "\n")


L = cfg.episode_length


def progress(step, m):
    # eval/episode_* are sums over the episode
    n = m['eval/avg_episode_length']
    log(f"[{time.time()-t0:6.0f}s] step {step:>11,}  reward {m['eval/episode_reward']:7.2f}  len {n:4.0f}"
        f"  path {m.get('eval/episode_progress', float('nan'))/max(n,1)*100:5.1f} cm(avg)"
        f"  xtrack {m.get('eval/episode_cross_track', float('nan'))/max(n,1)*1000:5.1f} mm"
        f"  at_goal {m.get('eval/episode_at_goal', float('nan'))/L*100:5.1f}%"
        f"  faulty {m.get('eval/episode_faulty', float('nan'))/max(n,1)*100:4.0f}%")


def save(step, make_policy, params):
    pickle.dump(jax.tree_util.tree_map(np.asarray, params), open(os.path.join(out, "params.pkl"), "wb"))


networks = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=POLICY_LAYERS,
    value_hidden_layer_sizes=VALUE_LAYERS,
    policy_obs_key="state",
    value_obs_key="privileged_state",
)
make_policy, params, _ = ppo.train(
    environment=env,
    num_timesteps=STEPS, num_evals=int(args.get("evals", 25)), reward_scaling=1.0, episode_length=L,
    normalize_observations=True, action_repeat=1, unroll_length=20, num_minibatches=32,
    num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2,
    num_envs=ENVS, batch_size=ENVS // 32, max_grad_norm=1.0, network_factory=networks,
    seed=int(args.get("seed", 0)), progress_fn=progress, policy_params_fn=save,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
save(STEPS, make_policy, params)
log(f"done in {time.time()-t0:.0f}s")
