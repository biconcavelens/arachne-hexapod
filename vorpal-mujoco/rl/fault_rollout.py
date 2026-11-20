"""
Shared by eval_fault.py, diagnose.py and render_fault.py: load a trained policy and
run batched episodes in fault_path_env with chosen (or random) faults.

A scenario fixes the fault: kind (index into KINDS), which servo, onset step, size.
servo = -1 means none; servo2 adds a second fault (same onset + 40 steps).
"""
import json
import os
import pickle

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

import fault_path_env as E

ROOT = os.path.dirname(os.path.abspath(__file__))

# mid-range fault sizes used in the evaluation scenarios
EVAL_SIZE = {0: 0.0, 1: 0.8, 2: float(np.radians(15)), 3: float(np.radians(3.5)), 4: 0.0, 5: 0.0}


def load_policy(name, env):
    run = os.path.join(ROOT, "runs", name)
    cfg = json.load(open(os.path.join(run, "config.json")))
    import room_scene as RS_
    want, have = cfg.get("cam", "24x16"), f"{RS_.CAM_W}x{RS_.CAM_H}"
    assert want == have, f"{name} was trained with a {want} camera; run with ARACHNE_CAM={want} (now {have})"
    params = pickle.load(open(os.path.join(run, "params.pkl"), "rb"))
    import policy_nets
    nets = policy_nets.factory(cfg.get("encoder", "mlp"), cfg.get("cam_shape"), tuple(cfg["policy_layers"]),
                               tuple(cfg["value_layers"]))(
        env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize)
    policy = ppo_networks.make_inference_fn(nets)((params[0], params[1]), deterministic=True)
    return lambda obs: policy(obs, jax.random.PRNGKey(0))[0]


def set_faults(state, kinds, servos, onsets, sizes):
    """kinds/servos/onsets/sizes: arrays of shape (n_faults,) for one env; servo -1 = unused."""
    kind = jp.zeros(12, jp.int32)
    size = jp.zeros(12)
    onset = jp.full(12, 10_000)
    for j in range(len(kinds)):
        ok = servos[j] >= 0
        idx = jp.maximum(servos[j], 0)
        kind = kind.at[idx].set(jp.where(ok, kinds[j], kind[idx]))
        size = size.at[idx].set(jp.where(ok, sizes[j], size[idx]))
        onset = onset.at[idx].set(jp.where(ok, onsets[j], onset[idx]))
    info = dict(state.info)
    info.update(fault_kind=kind, fault_size=size, fault_onset=onset)
    return state.replace(info=info)


def make_runner(env, policy, steps=None, record_obs=False, random_faults=False, goal_idx=None, record_mocap=False):
    """Returns run(rng_keys, kinds, servos, onsets, sizes) -> dict of per-step arrays (B, T, ...).
    With random_faults=True the env's own fault sampling is kept and the fault args are ignored.
    goal_idx overrides the sampled goal (path index; the path is N_PATH points long)."""
    T = steps or env._config.episode_length

    def one(key, kinds, servos, onsets, sizes):
        s = env.reset(key)
        if goal_idx is not None:
            info = dict(s.info)
            info["goal_idx"] = jp.asarray(goal_idx, jp.int32)
            s = s.replace(info=info)
        if not random_faults:
            s = set_faults(s, kinds, servos, onsets, sizes)

        def body(carry, _):
            s, done = carry
            obs_state = s.obs["state"]
            a = policy(s.obs)
            s2 = env.step(s, a)
            done2 = done | (s2.done > 0)
            rem = s2.info["goal_idx"] * E.DS - s2.info["s"]
            out = {
                "xy": s2.data.qpos[0:2], "yaw": env._yaw(s2.data), "qpos": s2.data.qpos,
                "s": s2.info["s"], "remaining": rem, "cross_track": s2.metrics["cross_track"],
                "done": done2, "fault_active": env.fault_active(s2.info),
            }
            if "heading_err" in s2.metrics:
                out["heading_err"] = s2.metrics["heading_err"]
            if "struck" in s2.metrics:
                out["struck"] = s2.metrics["struck"] > 0
            if record_mocap:
                out["mocap_pos"], out["mocap_quat"] = s2.data.mocap_pos, s2.data.mocap_quat
            if record_obs:
                out["obs"] = obs_state
                # what the robot itself knows whatever the policy's sensors: IMU and its last commands
                out["diag_x"] = jp.concatenate([env._gravity(s2.data), s2.data.qvel[3:6], s2.info["last_act"]])
                out["label"] = s.info["fault_kind"] * env.fault_active(s.info)
            return (s2, done2), out

        _, traj = jax.lax.scan(body, (s, jp.zeros((), bool)), None, length=T)
        traj["path"] = s.info["path"]
        traj["goal_idx"] = s.info["goal_idx"]
        traj["fault_kind"] = s.info["fault_kind"]
        traj["fault_onset"] = s.info["fault_onset"]
        return traj

    return jax.jit(jax.vmap(one))


def summarise(traj, dt):
    """Per-episode metrics from a batch of trajectories (numpy dict)."""
    done = traj["done"]
    fell = done[:, -1]
    reached = (traj["remaining"] < 0.03) & ~done
    goal = reached.any(1)
    t_goal = np.where(goal, reached.argmax(1) * dt, np.nan)
    alive = ~done
    xt = np.where(alive, traj["cross_track"], np.nan)
    path_len = traj["goal_idx"] * E.DS
    frac = np.clip(np.where(fell, traj["s"][np.arange(len(fell)), done.argmax(1)], traj["s"][:, -1]) / path_len, 0, 1)
    return {"goal": goal, "t_goal": t_goal, "fell": fell, "cross_track_mm": np.nanmean(xt, 1) * 1000,
            "max_cross_track_mm": np.nanmax(xt, 1) * 1000, "path_frac": frac, "path_len": path_len}
