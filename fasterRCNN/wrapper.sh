#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
conda activate faster-3.7
cd /mnt/h/jinhan/xai/fasterRCNNBDD

# Script to run another script, kill it after a fixed duration, and restart it

SCRIPT_PATH="/mnt/h/jinhan/xai/fasterRCNNBDD/multi_layer.sh"
RUN_TIME=60  # Time in seconds after which to kill the script

while true; do
    # Start the script in the background
    $SCRIPT_PATH &
    SCRIPT_PID=$!

    # Sleep for the duration the script should run
    sleep $RUN_TIME

    # Kill the script
    kill $SCRIPT_PID
    wait $SCRIPT_PID 2>/dev/null

    # Now kill any remaining processes that might be using the GPU
    # Loop through each GPU-using process and kill it
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read gpu_pid; do
        echo "Killing remaining GPU process PID: $gpu_pid"
        kill -9 $gpu_pid 2>/dev/null
    done

    # Optional: sleep before restarting or perform any checks
    #sleep 2
done