import numpy as np
import cv2, torch, pickle
from scipy import io
import os, logging, re

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(filename='/home/jinhanz/cs/xai/logs/241030_fasterrcnn_faithfulness_bilinear.log', 
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

# Define a lock to ensure thread-safe operations on shared arrays
lock = threading.Lock()

def process_image(img_idx, img_file, cur_sample, skip_images, root_path, dir, n, meanConf_deletionAI, meanConf_insertionAI):
    to_skip = [s for s in skip_images if s == img_file.split('-')[0]]
    if len(to_skip) > 0:
        return

    cur_sample = torch.load(os.path.join(os.path.join(root_path,dir),img_file))

    # Convert ground truth bounding boxes to appropriate format
    boxes_gt_xywh = cur_sample['boxes_gt_xywh']

    # Jinhan: sample N=10 steps from 100 steps

    # Deletion: 11 steps in total (intact image result at step 1 included)
    indices = list(range(9, 100, 10))
    perturb_steps_deletion = np.shape(cur_sample['preds_deletion'])

    if perturb_steps_deletion[1] == 10:
        intact_res = [
            cur_sample['boxes_gt_xyxy'], cur_sample['boxes_pred_xywh'], 
            [], [], torch.as_tensor(cur_sample['boxes_pred_conf']).numpy()
        ]
        if cur_sample['preds_deletion'].size == 0:
            cur_sample['preds_deletion'] = np.empty((5, 11), dtype=object)
        else:
            tmp = cur_sample['preds_deletion']
            if len(tmp.shape) == 3:  # DEBUG: Check for saving issue
                logging.warning(f'saving issue: {img_file}')
                tmp = tmp[:, :, 0]
            cur_sample['preds_deletion'] = np.empty((5, 11), dtype=object)
            cur_sample['preds_deletion'][:, 1:11] = tmp
        cur_sample['preds_deletion'][:, 0] = intact_res
    elif perturb_steps_deletion[1] == 100:
        cur_sample['preds_deletion'] = cur_sample['preds_deletion'][:, [0] + indices]

    perturb_steps_insertion = np.shape(cur_sample['preds_insertation'])
    if perturb_steps_insertion[1] == 100:
        cur_sample['preds_insertation'] = cur_sample['preds_insertation'][:, indices]

    if cur_sample['preds_deletion'].size == 0:
        cur_sample['preds_deletion'] = np.empty((5, n + 1), dtype=object)
    if cur_sample['preds_insertation'].size == 0:
        cur_sample['preds_insertation'] = np.empty((5, n), dtype=object)

    # Options Vector (Configuration Parameters)
    opt = {"BboxErr_Thr": 0, 
            "IoU_Thr": 0.5, 
            "ImageWidth": np.shape(cur_sample['masks_ndarray'])[1], 
            "ImageHeight": np.shape(cur_sample['masks_ndarray'])[0]}

    boxes_gt_xywh = cur_sample['boxes_gt_xywh']

    meanConf_deletionAI[:, img_idx] = getDeletionAI_res(cur_sample, boxes_gt_xywh, opt)
    meanConf_insertionAI[:, img_idx] = getInsertionAI_res(cur_sample, boxes_gt_xywh, opt)

# TODO: parallel
def compute_single_condition(root_path):  
    model = [m for m in ['fasterrcnn','yolov5s'] if m in root_path][0]
    dataset = [d for d in ['mscoco','_vehicle','_human'] if d in root_path][0].replace('_','')
    xai_method = [x for x in ['fullgradcamraw','odam'] if x in root_path][0]
    # rescale_method = [r for r in ['bilinear','sigma2','sigma4'] if r in root_path][0]
    is_act = True if f"_{model}_act_" in root_path else False

    logging.info(f"{model}_{dataset}_{xai_method}_optimize_faithfulnee: {'Activation Map' if is_act else 'Feature Map'}")

    if not os.path.exists(root_path): 
        logging.info('NOT STARTED')
        return
    
    skip_images = []
    if model == 'yolov5s':
        if dataset == 'mscoco':
            skip_images = ['book_472678',"baseball glove_515982","toothbrush_160666","potted plant_473219","bench_350607","truck_295420","toaster_232348","kite_405279","toothbrush_218439","snowboard_425906","car_227511","traffic light_453841","hair drier_239041","hair drier_178028","toaster_453302","mouse_513688","spoon_88040","scissors_340930","handbag_383842"]
        elif dataset == 'vehicle':
            skip_images = ["178", "54", "452", "478", "629", "758", "856",'1007', '1028', '1041', '1065', '1100', '1149', '1236', '1258', '1272', '1331', '1356', '210', '222', '3', '390', '431', '485', '505', '52', '559', '585', '634', '648', '670', '715', '784', '797', '803', '833', '848', '867', '899', '914', '940', '980', '993','1121', '1127', '1170', '1365', '321', '425', '542', '610', '896', '902', '953', '967']
        elif dataset == 'human':
            skip_images = ['1022', '1041', '1053', '1063', '1066', '1097', '11', '1141', '1142', '1154', '1227', '1228', '1273', '1293', '1302', '1313', '1346', '1359', '1398', '1420', '1430', '1475', '1506', '152', '1538', '1553', '1624', '1663', '1664', '1746', '1770', '1788', '1803', '1805', '1817', '1852', '186', '1863', '1893', '19', '1917', '1954', '2008', '2040', '2087', '2092', '2108', '2121', '2128', '2141', '2161', '2186', '2203', '2219', '2226', '2262', '2270', '2271', '2279', '231', '2312', '2327', '2334', '2457', '250', '286', '348', '388', '391', '415', '422', '425', '452', '47', '608', '670', '683', '748', '757', '805', '808', '829', '845', '85', '875', '897', '900', '928', '962', '97', '997']
    elif model == 'fasterrcnn':
        if dataset == 'mscoco':
            skip_images = ['book_472678','clock_164363','hair drier_178028','hair drier_239041', 'kite_405279', 'mouse_513688', 'toaster_232348', 'toaster_453302', 'toothbrush_218439', 'traffic light_453841']
        elif dataset == 'vehicle':
            skip_images = ['1007', '1023', '1028', '1041', '1079', '1108', '1121', '1127', '1170', '1201', '1253', '1258', '1272', '134', '1344', '1356', '210', '297', '321', '355', '383', '390', '406', '425', '485', '505', '52', '542', '634', '648', '711', '777', '784', '796', '797', '838', '848', '857', '899', '902', '953', '967', '969', '988', '99', '993']
        elif dataset == 'human':
            skip_images = ['2334', '1313', '1302', '2186', '1770', '1154', '1663', '186', '425', '875', '845', '829', '388', '748', '900', '1346', '1803', '1359', '1022', '97', '2203', '1066', '231', '1097', '488', '415', '2128', '2008', '2121', '2092', '2271', '1506', '1389', '1954', '2226', '670', '2161', '1041', '250', '1141', '348', '1063', '452', '601', '19', '1746', '1917', '1420', '1817', '270', '1398', '2040', '11', '1475', '897', '1805', '997', '1788']

    for dir in os.listdir(root_path):
        if not os.path.isdir(os.path.join(root_path,dir)): continue

        if model == 'fasterrcnn':
            try:
                layer_name = re.findall(r'_[a-zA-Z0-9]+\.[a-zA-Z0-9\.]+_',dir)[0].replace('_','')
            except:
                continue
        elif model == 'yolov5s':
            try:
                layer_name = re.findall(r'_model_[a-zA-Z0-9_]+_act_',dir)[0][1:-1]
            except:
                continue

        all_imgs = sorted(os.listdir(os.path.join(root_path,dir)))
        all_imgs = [f for f in all_imgs if '.pth' in f]

        cur_sample = None
        N = 10
        meanConf_deletionAI = np.zeros((N + 1, len(all_imgs)))
        meanConf_insertionAI = np.zeros((N, len(all_imgs)))

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(process_image, img_idx, img_file, cur_sample, skip_images, root_path, dir, N, meanConf_deletionAI, meanConf_insertionAI)
                for img_idx, img_file in enumerate(all_imgs)
            ]

            # Optionally, you can use `as_completed` to handle futures as they finish
            for future in as_completed(futures):
                try:
                    future.result()  # Check for any exceptions
                except Exception as e:
                    logging.error(f"Error processing image: {e}")
        
        # for img_idx, img_file in enumerate(all_imgs):
        #     process_image(img_idx, img_file, cur_sample, skip_images, root_path, dir, N, meanConf_deletionAI, meanConf_insertionAI)

        # At this point, you can save or process the results further
        # Example to save results
        results = {
            "meanConf_deletionAI": meanConf_deletionAI,
            "meanConf_insertionAI": meanConf_insertionAI,
            "all_imgs": all_imgs,
        }

        pickle.dump(results,open(os.path.join(root_path, f"{layer_name}.pickle"),'wb'))
        if os.path.exists(os.path.join(root_path, f"{layer_name}.npy")):
            os.remove(os.path.join(root_path, f"{layer_name}.npy"))
        logging.info(f"Layer {layer_name} finished at: {os.path.join(root_path, f'{layer_name}.pickle')}")

def xywh2xyxy(p):
    if p.size < 4:
        p = np.zeros((1, 4))
    
    p_new = np.zeros_like(p)
    p_new[0] = p[0] - p[2] / 2  # x_min
    p_new[1] = p[1] - p[3] / 2  # y_min
    p_new[2] = p[0] + p[2] / 2  # x_max
    p_new[3] = p[1] + p[3] / 2  # y_max

    return p_new

def xyxy2xywh(p):
    p_new = np.zeros_like(p)
    p_new[0] = (p[0] + p[2]) / 2  # center_x
    p_new[1] = (p[1] + p[3]) / 2  # center_y
    p_new[2] = p[2] - p[0]        # width
    p_new[3] = p[3] - p[1]        # height

    return p_new

def getDeletionAI_res(curSample, boxes_gt_xywh, opt_vec):
    meanConf_deletionAI = np.zeros(len(curSample['preds_deletion'][0]))

    for i in range(len(curSample['preds_deletion'][0])):
        if curSample['preds_deletion'][0][i] is None or\
            (type(np.asarray(curSample['preds_deletion'][0][i]).size) == int and np.asarray(curSample['preds_deletion'][0][i]).size == 0):
            meanConf_deletionAI[i] = 0 # no prediction
            continue
        pred_conf_list = np.atleast_1d(np.asarray(curSample['preds_deletion'][4][i]).squeeze())
        pred_corr_list = np.vstack(curSample['preds_deletion'][1][i])
        meanConf_deletionAI[i] = mean_valid_confidence(pred_corr_list, pred_conf_list, boxes_gt_xywh, opt_vec)

    return meanConf_deletionAI

def getInsertionAI_res(curSample, boxes_gt_xywh, opt_vec):
    meanConf_insertationAI = np.zeros(len(curSample['preds_insertation'][0]))

    for i in range(len(curSample['preds_insertation'][0])):
        if curSample['preds_insertation'][0][i] is None or\
            (type(np.asarray(curSample['preds_insertation'][0][i]).size) == int and np.asarray(curSample['preds_insertation'][0][i]).size == 0):
            meanConf_insertationAI[i] = 0 # no prediction
            continue
        pred_conf_list = np.atleast_1d(np.asarray(curSample['preds_insertation'][4][i]).squeeze())
        pred_corr_list = np.vstack(curSample['preds_insertation'][1][i])
        meanConf_insertationAI[i] = mean_valid_confidence(pred_corr_list, pred_conf_list, boxes_gt_xywh, opt_vec)

    return meanConf_insertationAI

def mean_valid_confidence(pred_corr_list, pred_conf_list, target_corr_list, opt):
    valid_confidence = np.zeros(target_corr_list.shape[0])
    for i, target_corr in enumerate(target_corr_list):
        max_iou = 0
        for j, pred_corr in enumerate(pred_corr_list):
            iou = IoU_cal(target_corr, pred_corr, opt)
            if iou > opt['IoU_Thr'] and iou > max_iou:
                valid_confidence[i] = pred_conf_list[j]
    return valid_confidence.mean()

def IoU_cal(curGT_Bbox_Corr, curPredictBbox, opt):
    H, W = opt['ImageHeight'], opt['ImageWidth']
    M_GT = np.zeros((H, W))
    M_PR = np.zeros((H, W))

    GTBbox_xyxy = np.round(xywh2xyxy(curGT_Bbox_Corr)).astype(int)
    PRBbox_xyxy = np.round(xywh2xyxy(curPredictBbox)).astype(int)

    GTBbox_xyxy = np.clip(GTBbox_xyxy, 1, [W, H, W, H])
    PRBbox_xyxy = np.clip(PRBbox_xyxy, 1, [W, H, W, H])

    M_GT[GTBbox_xyxy[1]:GTBbox_xyxy[3], GTBbox_xyxy[0]:GTBbox_xyxy[2]] = 1
    M_PR[PRBbox_xyxy[1]:PRBbox_xyxy[3], PRBbox_xyxy[0]:PRBbox_xyxy[2]] = 1

    IoU_rec = np.sum(M_GT * M_PR) / np.sum(np.logical_or(M_GT, M_PR))
    
    return IoU_rec

# Initial Setup
root_dir = "/opt/jinhanz/results/bilinear/"

for model in ['fasterrcnn']: #'yolov5s'
    for dataset in ['mscoco']:
        for is_act in ['']:
            for upscale_type in ['bilinear']:
                if dataset != 'mscoco':
                    xai_path = os.path.join(root_dir,'bdd',f"xai_saliency_maps_{model}")
                else:
                    xai_path = os.path.join(root_dir,dataset,f"xai_saliency_maps_{model}")

                for xai_method in ['fullgradcamraw','odam']:
                    if dataset == 'mscoco':
                        path = os.path.join(xai_path,xai_method)
                    else:
                        path = os.path.join(xai_path,f"{xai_method}_{dataset}")

                    compute_single_condition(path)