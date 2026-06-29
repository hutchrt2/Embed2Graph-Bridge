#!/usr/bin/env bash
# run_with_monitor.sh
# Usage: ./run_with_monitor.sh <command> [args]
# This script runs the given command, logs its stdout/stderr, and periodically records CPU and memory usage.

LOG_FILE="monitor_$(date +%Y%m%d_%H%M%S).log"

echo "Monitoring started at $(date)" > "$LOG_FILE"

# Start the command in background, tee output to the log
"$@" > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2) &
CMD_PID=$!

echo "Command PID: $CMD_PID" >> "$LOG_FILE"

# Monitor loop: log every 30 seconds while the process runs
while kill -0 $CMD_PID 2>/dev/null; do
    echo "$(date +'%Y-%m-%d %H:%M:%S') CPU/MEM:" >> "$LOG_FILE"
    ps -p $CMD_PID -o %cpu,%mem,rss,vsz >> "$LOG_FILE"
    sleep 30
done

echo "Command finished at $(date)" >> "$LOG_FILE"
