#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
conda activate xai-3.8
cd /home/jinhanz/cs/xai/yolov5COCO

for category in COCO vehicle human; do

while true; do
    # Run your program
    python main_faithful_cal_adaptive_yolo_optimize_faithfulness_apply_act.py --object $category --img-start 64 --img-end 96 --device 2
    
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