import os, mujoco_playground
root = os.path.join(os.path.dirname(mujoco_playground.__file__), "_src")
print(root)
for p in ["mjx_env.py", "locomotion/go1/base.py", "locomotion/go1/joystick.py", "locomotion/params.py"]:
    f = os.path.join(root, p)
    print("=" * 20, p, os.path.exists(f), sum(1 for _ in open(f)) if os.path.exists(f) else "")
print(sorted(os.listdir(os.path.join(root, "locomotion"))))
