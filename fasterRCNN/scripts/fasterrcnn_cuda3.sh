#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd /home/jinhanz/cs/xai/fasterRCNN

for category in COCO vehicle human; do

while true; do
    # Run your program
    python main_faith_adaptive_detect_general_optimize_faithfulness.py --object $category --img-start 96 --img-end 128 --device 3
    
    # Check the exit status
    if [ $? -ne 0 ]; then
        echo "Program exited with an error. Restarting in 10 seconds..."
        sleep 10
    else
        # If program exits successfully, break the loop
        echo "Program completed successfully."
        break
    fi
done

done