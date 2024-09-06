#! /bin/bash

# export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
# conda activate faster-3.7

#cd /mnt/h/jinhan/xai/fasterRCNN
python main_faith_adaptive_detect_odam_novalidarea.py
python main_faith_adaptive_detect_whole_image_novalidarea.py
#echo main_faith_adaptive_detect_bdd_whole_image.py
# python main_faith_adaptive_detect_bdd_whole_image.py
#echo main_faith_adaptive_detect_bdd_odam.py
# python main_faith_adaptive_detect_bdd_odam.py
