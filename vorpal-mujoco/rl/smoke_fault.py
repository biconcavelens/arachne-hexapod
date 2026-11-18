"""Smoke test for fault_path_env: compiles, stands, faults act as specified, throughput."""
import time

import jax
import jax.numpy as jp
import numpy as np

import common as C
from fault_path_env import FaultPath, KINDS

env = FaultPath()
print("obs", {k: v for k, v in env.observation_size.items()}, "act", env.action_size)
reset, step = jax.jit(env.reset), jax.jit(env.step)
stand = jp.asarray(C.STAND_ACTION)

# one servo of each kind, onset at step 10; command a hip sweep and see what each holds
for k in range(1, 6):
    s = reset(jax.random.PRNGKey(k))
    kind = jp.zeros(12, jp.int32).at[0].set(k)          # hip0
    size = jp.zeros(12).at[0].set([0, 0.9, np.radians(15), np.radians(3), 0, 0][k])
    s.info.update(fault_kind=kind, fault_size=size, fault_onset=jp.full(12, 10))
    qs = []
    for i in range(40):
        a = stand.at[0].set(0.6 * np.sin(i / 4))            # sweep hip0 by +-15 deg
        s = step(s, a)
        qs.append(float(s.data.qpos[7]))
    qs = np.degrees(qs)
    print(f"{KINDS[k]:9s} hip0 range before onset {np.ptp(qs[:10]):5.1f} deg, after {np.ptp(qs[15:]):5.1f} deg, "
          f"z {float(s.data.qpos[2])*1000:.0f} mm, done {float(s.done)}")

s = reset(jax.random.PRNGKey(0))
t0 = time.time()
for i in range(100):
    s = step(s, stand)
jax.block_until_ready(s.data.qpos)
print(f"standing 5 s: z {float(s.data.qpos[2])*1000:.1f} mm, done {float(s.done)}, reward {float(s.reward):.3f}, "
      f"goal {int(s.info['goal_idx'])} cm, faults {np.array(s.info['fault_kind'])}")

B = 4096
vreset, vstep = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
S = vreset(jax.random.split(jax.random.PRNGKey(1), B))
A = jp.tile(stand, (B, 1))
S = vstep(S, A)
jax.block_until_ready(S.data.qpos)
t0 = time.time()
for _ in range(50):
    S = vstep(S, A)
jax.block_until_ready(S.data.qpos)
dt = time.time() - t0
nf = np.array((S.info["fault_kind"] > 0).sum(1))
print(f"{B} envs: {50*B/dt:,.0f} env-steps/s; faulty servos per episode: "
      f"{np.bincount(nf, minlength=3) / B}")
