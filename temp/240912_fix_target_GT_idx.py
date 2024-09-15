import os,re
import numpy as np
from scipy import io

root_dir = '/mnt/h/jinhan/results/'

wrong_layer = 'model_1_act'
correct_layer= 'model_21_act'

for model in ['yolov5s']:
    for dataset in ['mscoco']:
        for upscale_type in ['bilinear']:
            if dataset != 'mscoco':
                xai_path = os.path.join(root_dir,'bdd',f"xai_saliency_maps_{model}_{upscale_type}")
            else:
                xai_path = os.path.join(root_dir,dataset,f"xai_saliency_maps_{model}_{upscale_type}")
            for xai_method in ["fullgradcamraw",'odam']:
                if dataset == 'mscoco':
                    path = os.path.join(xai_path,xai_method)
                else:
                    path = os.path.join(xai_path,f"{xai_method}_{dataset}")

                for dir in os.listdir(path):
                    if any([l for l in ["model_21","model_23"] if l in dir]):
                        continue
                    if not os.path.isdir(os.path.join(path,dir)): continue

                    for img_file in os.listdir(os.path.join(path,dir)):

                        img_name = img_file.replace('-res.png','').replace('.mat','')

                        if img_name not in ['chair_81061','elephant_97230','giraffe_287545','airplane_167540', 'airplane_338325', 'apple_216277', 'apple_562059', 'backpack_177065', 'backpack_370478', 'banana_279769', 'banana_290619', 'baseball bat_129945', 'baseball bat_270474', 'baseball glove_162415', 'baseball glove_515982', 'bear_519611', 'bear_521231', 'bed_468245', 'bed_491757', 'bench_310072', 'bench_350607', 'bicycle_203317', 'bicycle_426166', 'bird_100489', 'bird_404568', 'boat_178744', 'boat_442822', 'book_167159', 'book_472678', 'bottle_385029', 'bottle_460929', 'bowl_205834', 'bowl_578871', 'broccoli_389381', 'broccoli_61658']:
                            os.remove(os.path.join(path,dir,img_file))

                    # os.remove(os.path.join(path,dir,'giraffe_287545-res.png'))
                    # os.remove(os.path.join(path,dir,'giraffe_287545-res.png.mat'))

                # wrong_layer_dir = os.path.join(path,f"{xai_method}_COCO_NMS_class_{wrong_layer}_aifaith_norm_yolov5s_COCOPretrained_1")
                # correct_layer_dir = os.path.join(path,f"{xai_method}_COCO_NMS_class_{correct_layer}_aifaith_norm_yolov5s_COCOPretrained_1")

                # for img_file in os.listdir(wrong_layer_dir):
                #     if '.mat' not in img_file: continue
                #     wrong_mat = io.loadmat(os.path.join(wrong_layer_dir,img_file))
                #     correct_mat = io.loadmat(os.path.join(correct_layer_dir,img_file))

                #     if not np.array_equal(wrong_mat['boxes_gt_xywh'],correct_mat['boxes_gt_xywh']):
                #         pass
