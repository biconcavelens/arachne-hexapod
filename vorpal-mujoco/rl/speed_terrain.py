"""Env throughput and compile check:  python speed_terrain.py [envs] [sensors] [scene]"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import jax, jax.numpy as jp
import scenes
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
SENS = sys.argv[2] if len(sys.argv) > 2 else "both"
SCENE = sys.argv[3] if len(sys.argv) > 3 else "terrain"
env = scenes.env_class(SCENE)(config_overrides={"sensors": SENS})
vr, vs = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
t0 = time.time()
st = vr(jax.random.split(jax.random.PRNGKey(0), N)); a = jp.zeros((N, 12))
st = vs(st, a); jax.block_until_ready(st.obs["state"]); print(f"compile {time.time() - t0:.0f}s", flush=True); t0 = time.time()
for _ in range(10): st = vs(st, a)
jax.block_until_ready(st.obs["state"])
print(f"{SCENE} {SENS}: {N * 10 / (time.time() - t0):,.0f} env-steps/s at {N} envs", "done", float(st.done.mean()))
