import os, shutil
import scipy.io

"""
Remove the previously saved masks_array_all item to save HKU workstation space
"""

root_dirs = [
    '/mnt/h/jinhan/results/ust/mscoco/xai_saliency_maps_yolov5s_bilinear',
    '/mnt/h/jinhan/results/ust/mscoco/xai_saliency_maps_yolov5s_gaussian_sigma2',
    '/mnt/h/jinhan/results/ust/mscoco/xai_saliency_maps_yolov5s_gaussian_sigma4',
    '/mnt/h/jinhan/results/mscoco/xai_saliency_maps_yolov5s_bilinear',
    '/mnt/h/jinhan/results/mscoco/xai_saliency_maps_yolov5s_gaussian_sigma2',
    '/mnt/h/jinhan/results/mscoco/xai_saliency_maps_yolov5s_gaussian_sigma4',
    ]

for root in root_dirs:
    for xai_method in ['fullgradcamraw','odam']:
        xai_path = os.path.join(root,xai_method)
        for layer_dir in os.listdir(xai_path):
            target_layer_dir = os.path.join(xai_path,layer_dir)
            if not os.path.isdir(target_layer_dir):
                continue
            for img_file in os.listdir(target_layer_dir):
                if '.mat' not in img_file: continue
                mat = scipy.io.loadmat(os.path.join(target_layer_dir,img_file))

                if 'masks_ndarray_all' in mat:
                    del mat['masks_ndarray_all']
                    scipy.io.savemat(os.path.join(target_layer_dir,img_file),mat)

                


