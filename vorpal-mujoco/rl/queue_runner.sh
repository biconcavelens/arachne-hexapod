#!/usr/bin/env bash
# Runs GPU jobs one at a time from runs/queue.txt (one shell command per line, run in this folder).
# Append lines any time; the first line is removed when its job starts. Start it as a user unit:
#   systemd-run --user --unit=arachne-queue --working-directory="$PWD" bash queue_runner.sh
# Each job runs in its own unit (arachne-job) with its own memory cap, so an out-of-memory kill ends
# only that job and the queue goes on. Default cap 7G; a line may start with "@mem=9G " to change it.
# Waits first for any unit named in WAIT_FOR (space-separated) to finish.
cd "$(dirname "$0")"
Q=runs/queue.txt
LOG=runs/queue.log
touch "$Q"
for u in ${WAIT_FOR:-}; do
  while systemctl --user is-active -q "$u"; do sleep 30; done
done
while true; do
  line=$(head -n 1 "$Q")
  if [ -z "$line" ]; then sleep 60; continue; fi
  sed -i '1d' "$Q"
  mem=7G
  if [[ "$line" =~ ^@mem=([0-9.]+[GM])\ (.*)$ ]]; then mem=${BASH_REMATCH[1]}; line=${BASH_REMATCH[2]}; fi
  echo "[$(date '+%F %T')] start (cap $mem): $line" >> "$LOG"
  systemd-run --user --unit=arachne-job --wait --collect --quiet -p MemoryMax="$mem" -p MemorySwapMax=0 \
    --working-directory="$PWD" bash -c "$line"
  rc=$?
  echo "[$(date '+%F %T')] exit $rc: $line" >> "$LOG"
done
