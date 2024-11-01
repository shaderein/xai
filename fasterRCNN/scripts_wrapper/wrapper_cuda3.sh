#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
conda activate faster-3.7
cd /home/jinhanz/cs/xai/fasterRCNN

# Script to run another script, kill it after a fixed duration, and restart it

SCRIPT_PATH="/home/jinhanz/cs/xai/fasterRCNN/multi_layer_cuda3.sh"
RUN_TIME=1100  # Time in seconds after which to kill the script

while true; do
    # Start the script in the background
    $SCRIPT_PATH &
    SCRIPT_PID=$!

    # Sleep for the duration the script should run
    sleep $RUN_TIME

    child_pids=$(pgrep -P $SCRIPT_PID)

    # Kill the script
    kill $SCRIPT_PID
    wait $SCRIPT_PID 2>/dev/null

    for child_pid in $child_pids; do
        # Kill the child process
        echo "Killing child process PID: $child_pid"
        kill -9 $child_pid 2>/dev/null
    done

    # Optional: sleep before restarting or perform any checks
    #sleep 2
done
