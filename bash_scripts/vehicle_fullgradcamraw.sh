#! /bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
conda activate xai-3.8
cd /mnt/h/jinhan/xai/yolov5BDD_samelayer
# cd /mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1

#python -W ignore main_faithful_cal_adaptive_yolo_BDD_whole_image.py --object human --model-path /mnt/h/jinhan/xai/models/yolov5sbdd100k300epoch.pt --method fullgradcamraw --prob class --output-main-dir "/mnt/h/OneDrive - The University Of Hong Kong/bdd/xai_saliency_maps_same_layer/fullgradcamraw_human" --img-path "/mnt/h/OneDrive - The University Of Hong Kong/bdd/images/orib_hum_id_task1009" --label-path "/mnt/h/OneDrive - The University Of Hong Kong/bdd/images/orib_hum_id_task1009_label"
python -W ignore main_faithful_cal_adaptive_yolo_BDD_whole_image.py --object vehicle --model-path /mnt/h/jinhan/xai/models/yolov5sbdd100k300epoch.pt --method fullgradcamraw --prob class --output-main-dir "/mnt/h/OneDrive - The University Of Hong Kong/bdd/xai_saliency_maps_same_layer/fullgradcamraw_vehicle" --img-path "/mnt/h/OneDrive - The University Of Hong Kong/bdd/images/orib_veh_id_task0922" --label-path "/mnt/h/OneDrive - The University Of Hong Kong/bdd/images/orib_veh_id_task0922_label"
