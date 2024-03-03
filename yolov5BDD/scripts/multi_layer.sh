#! /bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
conda activate xai-3.8
cd /mnt/h/jinhan/xai
# cd /mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1
for prob in class #obj objclass
do 
    for method in odam #gradcam fullgradcam fullgradcamraw gradcampp 
    do
        for object in human
        do
            img_path='orib_hum_id_task1009'
            label_path='orib_hum_id_task1009_label'
            echo "python -W ignore -m scripts.main_faithful_cal_adaptive_wsl.py --object $object --method $method --prob $prob --img-path $img_path --label-path $label_path"
            python -W ignore -m scripts.main_faithful_cal_adaptive_wsl.py --object $object --method $method --prob $prob --img-path $img_path --label-path $label_path
        done
    done
done