"""Smoke test: env compiles, steps, stands with the stand action, and how fast is it?"""
import time

import jax
import jax.numpy as jp
import numpy as np

import common as C
from hexapod_env import Hexapod

env = Hexapod()
print("obs size", env.observation_size, "action size", env.action_size, "dt", env.dt, "substeps", env.n_substeps)
reset, step = jax.jit(env.reset), jax.jit(env.step)
s = reset(jax.random.PRNGKey(0))
stand = jp.asarray(C.STAND_ACTION)
t0 = time.time()
for i in range(100):                                    # 5 s of standing still
    s = step(s, stand)
jax.block_until_ready(s.data.qpos)
print(f"compile+100 steps {time.time()-t0:.1f}s")
print(f"after 5 s standing: z {float(s.data.qpos[2])*1000:.1f} mm, done {float(s.done)}, "
      f"grav {np.round(np.array(s.obs['state'][:3]), 2)}, reward {float(s.reward):.3f}")
print("contacts", np.array(s.info["last_contact"]).astype(int), "command", np.round(np.array(s.info["command"]), 3))

# throughput, batched
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
print(f"{B} envs: {50*B/dt:,.0f} env-steps/s ({50*B/dt*env.dt:,.0f} sim-s per s)")
