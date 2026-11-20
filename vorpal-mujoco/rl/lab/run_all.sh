#!/usr/bin/env bash
# Runs the whole experiment on the lab machine: 12 trainings, evaluations, diagnosis, videos.
# Resumable: a step whose output already exists is skipped, so after a crash or reboot just start it
# again. After every step the results are committed and pushed to the branch lab-results.
#
#   cd vorpal-mujoco/rl
#   setsid nohup bash lab/run_all.sh > lab/run_all.log 2>&1 &
#
# Settings (environment variables):
#   ARACHNE_CAM   body-camera resolution, WxH (default 96x64; the laptop used 24x16)
#   ENVS          parallel robots per training (default 4096)
#   STEPS         training steps per policy (default 40000000)
#   PUSH          1 = commit and push results after each step (default 1)
set -u
cd "$(dirname "$0")/.."
export ARACHNE_CAM=${ARACHNE_CAM:-96x64}
ENVS=${ENVS:-4096}
STEPS=${STEPS:-40000000}
PUSH=${PUSH:-1}
P=${PREFIX:-v3_}                        # run-name prefix: heading fix + CNN camera encoder (20 Nov 2026)
EVAL_N=${EVAL_N:-256}                   # episodes per evaluation scenario
DIAG_EP=${DIAG_EP:-1536}                # rollouts for each fault detector
LOGS=lab/logs
mkdir -p $LOGS
say() { echo "[$(date '+%F %T')] $*" | tee -a lab/progress.txt; }
py() { bash wslrun.sh python "$@"; }
push() { [ "$PUSH" = 1 ] && bash lab/push_results.sh "$1" >> $LOGS/push.log 2>&1; return 0; }

# step <name> <done-check-file> <command...>: run unless done, log, push
step() {
  local name=$1 check=$2; shift 2
  if [ -e "$check" ]; then say "skip $name (done)"; return 0; fi
  say "start $name"
  local t0=$SECONDS
  "$@" > $LOGS/$name.out 2>&1
  local rc=$?
  say "end $name: exit $rc after $(( (SECONDS - t0) / 60 )) min"
  [ $rc -ne 0 ] && tail -n 20 $LOGS/$name.out | tee -a lab/progress.txt
  push "$name (exit $rc)"
  return 0
}
trained() { echo "runs/$P$1/DONE"; }
train() {   # train <name> <env> <sensors>
  py train_fault.py name=$P$1 env=$2 sensors=$3 steps=$STEPS envs=$ENVS && touch "$(trained $1)"
}

say "run_all: camera $ARACHNE_CAM, $ENVS envs, $STEPS steps, GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# 0. preflight: speed and memory at this resolution (fails fast if it does not fit)
step preflight $LOGS/preflight.ok bash -c "
  bash wslrun.sh python speed_terrain.py $ENVS both terrain_dyn &&
  bash wslrun.sh python train_fault.py name=${P}preflight env=terrain_dyn sensors=both steps=2000000 envs=$ENVS evals=2 &&
  touch $LOGS/preflight.ok"
if [ ! -e $LOGS/preflight.ok ]; then
  say "PREFLIGHT FAILED at $ARACHNE_CAM - see $LOGS/preflight.out. If it ran out of GPU memory, rerun with ARACHNE_CAM=72x48."
  exit 1
fi
# time estimate from the preflight training (step 0 -> last evaluation, compile excluded)
python3 - "$P" "$STEPS" << 'EOF' | tee -a lab/progress.txt
import re, sys
rows = [(float(t), int(s.replace(",", ""))) for t, s in
        re.findall(r"\[\s*(\d+)s\] step\s+([\d,]+)", open(f"runs/{sys.argv[1]}preflight/log.txt").read())]
(t0, s0), (t1, s1) = rows[0], rows[-1]
rate = (s1 - s0) / max(t1 - t0, 1)
h = 8 * int(sys.argv[2]) / rate / 3600          # 8 trainings with the camera; the 4 IMU-only ones are much faster
print(f"preflight: {rate:,.0f} training steps/s with the camera -> about {h:.0f} h for the 8 camera trainings "
      f"(+ roughly {h / 5:.0f} h for IMU-only trainings, evaluations and videos)")
EOF

# 1. training: four stages x three sensor sets
for st in "room room" "dyn room_dyn" "terr terrain" "tdyn terrain_dyn"; do
  set -- $st
  for s in imu both cam; do
    step train_$1_$s "$(trained $1_$s)" train $1_$s $2 $s
  done
done

# 2. evaluation, 256 episodes per scenario; mover and terrain stages also test the easier-stage policies
ev() { local out=$1; shift; step $out runs/$P$out/results.json py eval_room.py "$@" n=$EVAL_N out=$P$out; }
ev eval_room policies=${P}room_both,${P}room_imu,${P}room_cam env=room
ev eval_dyn  policies=${P}dyn_both,${P}dyn_imu,${P}dyn_cam,${P}room_both env=room_dyn
ev eval_terr policies=${P}terr_both,${P}terr_imu,${P}terr_cam,${P}room_both env=terrain
ev eval_tdyn policies=${P}tdyn_both,${P}tdyn_imu,${P}tdyn_cam,${P}terr_both,${P}room_both env=terrain_dyn

# 3. which servo is broken (IMU + commands), one detector per stage
for p in room_both dyn_both terr_both tdyn_both; do
  step diag_$p runs/${P}diag_$p/results.json py diagnose_room.py policy=$P$p out=${P}diag_$p episodes=$DIAG_EP
done

# 4. videos (GIF + small MP4), same episodes as the 20 Nov report
V=media/lab${PREFIX:+_$PREFIX}
mkdir -p $V
gif() { local out=$1; shift; step gif_$out $V/$out.mp4 bash -c "bash wslrun.sh python render_room.py $* out=$V/$out.gif && bash wslrun.sh python lab/to_mp4.py $V/$out.gif"; }
gif room_both_dead_stuck  policy=${P}room_both env=room kind=dead,stuck servo=knee1,hip4 onset=40 diag=${P}diag_room_both seed=1
gif room_cam_dead_stuck   policy=${P}room_cam  env=room kind=dead,stuck servo=knee1,hip4 onset=40 seed=1
gif dyn_both_dead         policy=${P}dyn_both  env=room_dyn kind=dead servo=knee2 onset=40 seed=2
gif terr_both_3mixed      policy=${P}terr_both env=terrain kind=dead,stuck,signal servo=knee1,hip3,knee5 onset=0 seed=3
gif terr_2dead_terrpolicy policy=${P}terr_both env=terrain kind=dead servo=knee1,knee4 onset=40 seed=4
gif terr_2dead_roompolicy policy=${P}room_both env=terrain kind=dead servo=knee1,knee4 onset=40 seed=4
gif tdyn_both_healthy     policy=${P}tdyn_both env=terrain_dyn kind=none seed=6
gif tdyn_both_dead_worn   policy=${P}tdyn_both env=terrain_dyn kind=dead,worn servo=knee2,hip5 onset=40 diag=${P}diag_tdyn_both seed=5 seconds=30
gif tdyn_cam_dead_worn    policy=${P}tdyn_cam  env=terrain_dyn kind=dead,worn servo=knee2,hip5 onset=40 seed=5 seconds=30

say "ALL DONE"
push "all done"
