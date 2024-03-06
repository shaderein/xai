% Author: Guoyang Liu.
% Date: 10/6/2022
% Function: Hit Rate Calculation
% Email: gyangliu@hku.hk

clear;

addpath(genpath(fullfile(pwd, 'utils')));


Path.resultBasePath = 'H:\Projects\HKU_XAI_Project\Yolov5self_GradCAM_Pytorch_1\update1_Stimuli_Images_Backward_Labels_Results';

% Method_Layer = {'gradcam_F1'};
% Method_Layer = {'gradcampp_F1'};

% Method_Layer = {'fullgradcam_F1'; 'fullgradcam_F2';'fullgradcam_F3';'fullgradcam_F4';'fullgradcam_F5';'fullgradcam_F6';
%     'fullgradcam_F7';'fullgradcam_F8';'fullgradcam_F9';'fullgradcam_F10';'fullgradcam_F11';'fullgradcam_F12';
%     'fullgradcam_F13'; 'fullgradcam_F14'; 'fullgradcam_F15'; 'fullgradcam_F16'; 'fullgradcam_F17'};

% Method_Layer = {'fullgradcamraw_F1'; 'fullgradcamraw_F2';'fullgradcamraw_F3';'fullgradcamraw_F4';'fullgradcamraw_F5';'fullgradcamraw_F6';
%     'fullgradcamraw_F7';'fullgradcamraw_F8';'fullgradcamraw_F9';'fullgradcamraw_F10';'fullgradcamraw_F11';'fullgradcamraw_F12';
%     'fullgradcamraw_F13'; 'fullgradcamraw_F14'; 'fullgradcamraw_F15'; 'fullgradcamraw_F16'; 'fullgradcamraw_F17'};

for iStrategy = 1:1
%     cur_Method_Layer = Method_Layer{iStrategy};
%     curMethodLayer = split(cur_Method_Layer, '_');
%     curMethod = curMethodLayer{1};
%     curLayer = curMethodLayer{2};
    curFolderName = 'human_saliency_map_backward_full_mat';
    Path.curSaliencyMapDataPath = fullfile(Path.resultBasePath, curFolderName);
    Path.humanSaliencyMapDataPath = ['H:\Projects\HKU_XAI_Project\ToM_Project_1\ToM Compiled Data\Data\' curFolderName];
    allDir = dir(fullfile(Path.curSaliencyMapDataPath,"*.mat"));
    imgDataPath = 'H:\Projects\HKU_XAI_Project\Yolov5self_GradCAM_Pytorch_1\Stimuli_Images_Backward';
    N = 100;
    meanConf_deletionAI = zeros(N, numel(allDir));
    meanConf_insertationAI = zeros(N, numel(allDir));
    meanConf_deletionHuman = zeros(N, numel(allDir));
    meanConf_insertationHuman = zeros(N, numel(allDir));
    AUC_rec = zeros(numel(allDir), 1);
    AUC_human_rec = zeros(numel(allDir), 1);

    %%
    for iSample = 1:numel(allDir)   % parfor
        tic;
        curSample = load(fullfile(allDir(iSample).folder, allDir(iSample).name));
        if isempty(curSample.preds_deletion)
            curSample.preds_deletion = cell(5, N);
        end
        if isempty(curSample.preds_insertation)
            curSample.preds_insertation = cell(5, N);
        end
        if isempty(curSample.human_deletion)
            curSample.human_deletion = cell(5, N);
        end
        if isempty(curSample.human_insertation)
            curSample.human_insertation = cell(5, N);
        end
    
        % ************ This section compensate for the deletion cell like
        % 5x100x3 *********
        if size(curSample.human_deletion, 3) > 1
            warning('>5x100');
            curSample.human_deletion = combineFaithCell(curSample.human_deletion);
        end
        if size(curSample.human_insertation, 3) > 1
            warning('>5x100');
            curSample.human_insertation = combineFaithCell(curSample.human_insertation);
        end
        % ************************************************************* %

        fileNum = split(allDir(iSample).name, '-');
        curRawImg = imread(fullfile(imgDataPath, [fileNum{1} '.jpg']));
    
        % AUC
        H = size(curSample.masks_ndarray, 1);
        W = size(curSample.masks_ndarray, 2);
        maskArray = double(curSample.masks_ndarray);
        labels = zeros(H, W);
        for k = 1:size(curSample.boxes_gt_xyxy, 1)
            curXY = round(curSample.boxes_gt_xyxy(k, :));
            curXY(3) = min(curXY(3),W);
            curXY(4) = min(curXY(4),H);
            curXY(curXY<1) = 1;
            labels(curXY(2):curXY(4), curXY(1):curXY(3)) = 1;
        end
        Loc_T = zeros(100,1);
        thrArray = 0.01:0.01:1;
        for iCnt = 1:numel(thrArray)
            iT = thrArray(iCnt);
            Loc_T(iCnt) = sum((maskArray>iT).*labels,'all')./(sum(labels,'all')+sum((maskArray>iT).*(~labels),'all'));
        end
        AUC = mean(Loc_T);

%         [X,Y,T,AUC] = perfcurve(labels(:),maskArray(:), 1);
        AUC_rec(iSample) = AUC;
    
        % Human Saliency Map AUC
        humanSaliencyMap = load(fullfile(Path.humanSaliencyMapDataPath, [fileNum{1} '.jpg.mat']));
        humanSaliencyMap = humanSaliencyMap.att_smo;
        maskArray = double(humanSaliencyMap);
        labels = zeros(H, W);
        for k = 1:size(curSample.boxes_gt_xyxy, 1)
            curXY = round(curSample.boxes_gt_xyxy(k, :));
            curXY(3) = min(curXY(3),W);
            curXY(4) = min(curXY(4),H);
            curXY(curXY<1) = 1;
            labels(curXY(2):curXY(4), curXY(1):curXY(3)) = 1;
        end
        Loc_T = zeros(100,1);
        thrArray = 0.01:0.01:1;
        for iCnt = 1:numel(thrArray) % Jinhan: what performance curve?
            iT = thrArray(iCnt);
            Loc_T(iCnt) = sum((maskArray>iT).*labels,'all')./(sum(labels,'all')+sum((maskArray>iT).*(~labels),'all'));
        end
        AUC = mean(Loc_T);
%         [X,Y,T,AUC] = perfcurve(labels(:),maskArray(:), 1);
        AUC_human_rec(iSample) = AUC;
    
        % Deletion & Insertion
        opt_BboxErr_Thr = 0;
        opt_IoU_Thr = 0.5;
        opt_ImageWidth = size(curSample.masks_ndarray,2);
        opt_ImageHeight = size(curSample.masks_ndarray,1);
        opt_vec = [opt_BboxErr_Thr opt_IoU_Thr opt_ImageWidth opt_ImageHeight];
        
    
    %     curSample.boxes_pred_xywh = cat(2, curSample.boxes_pred_xywh, curSample.boxes_pred_conf);   % Combine Box Position and Confidence
        if numel(curSample.boxes_pred_xywh)<4
            curSample.boxes_pred_xywhc_baseline = [];
        else
            curSample.boxes_pred_xywhc_baseline = cat(2, curSample.boxes_pred_xywh, curSample.boxes_pred_conf);
        end
        boxes_gt_xywh = curSample.boxes_gt_xywh;
    
        meanConf_deletionAI(:, iSample) = getDeletionAI_res(curSample, boxes_gt_xywh, opt_vec);
        meanConf_insertationAI(:, iSample) = getInsertionAI_res(curSample, boxes_gt_xywh, opt_vec);
        meanConf_deletionHuman(:, iSample) = getDeletionHuman_res(curSample, boxes_gt_xywh, opt_vec);
        meanConf_insertationHuman(:, iSample) = getInsertationHuman_res(curSample, boxes_gt_xywh, opt_vec);
    
        disp(['Processing: ' num2str(iSample) '/' num2str(numel(allDir))]);
        toc;
    end
    meanAUC(:,iStrategy) = mean(AUC_rec, 'omitnan');
    meanHumanAUC(:,iStrategy) = mean(AUC_human_rec, 'omitnan');
    meanConf_deletionAI_all(:,:,iStrategy) = meanConf_deletionAI;
    meanConf_insertationAI_all(:,:,iStrategy) = meanConf_insertationAI;
    meanConf_deletionHuman_all(:,:,iStrategy) = meanConf_deletionHuman;
    meanConf_insertationHuman_all(:,:,iStrategy) = meanConf_insertationHuman;


end

% figure;
% subplot(1,2,1);
% imagesc(squeeze(mean(meanConf_deletionAI_all,2, 'omitnan')));
% subplot(1,2,2);
% imagesc(squeeze(mean(meanConf_insertationAI_all,2, 'omitnan')));
% colormap('jet')
% 
% figure;
% subplot(1,2,1);
% bar(squeeze(mean(meanConf_deletionAI_all,[1 2], 'omitnan')));
% subplot(1,2,2);
% bar(squeeze(mean(meanConf_insertationAI_all,[1 2], 'omitnan')));
% 
% figure;
% bar((meanAUC));

save(fullfile(pwd, [curFolderName '_faithfulness_rawConf_update1.mat']));

%% Support Functions
function human_deletion_combine = combineFaithCell(human_deletion)

human_deletion_combine = cell(size(human_deletion, 1), size(human_deletion, 2));
for jj = 1:size(human_deletion, 2)
    c1 = [];
    c2 = [];
    c3 = [];
    c4 = [];
    c5 = [];
    for kk = 1:size(human_deletion, 3)
        c1 = cat(1, c1, human_deletion{1, jj, kk});
        c2 = cat(1, c2, human_deletion{2, jj, kk});
        c3 = cat(2, c3, human_deletion{3, jj, kk});
        c4 = cat(2, c4, human_deletion{4, jj, kk});
        c5 = cat(2, c5, human_deletion{5, jj, kk});
    end
    human_deletion_combine{1, jj} = c1;
    human_deletion_combine{2, jj} = c2;
    human_deletion_combine{3, jj} = c3;
    human_deletion_combine{4, jj} = c4;
    human_deletion_combine{5, jj} = c5;
end

end

function meanConf_deletionAI = getDeletionAI_res(curSample, boxes_gt_xywh, opt_vec)
    % Deletion: AI
    meanConf_deletionAI = zeros(size(curSample.preds_deletion, 2), 1);
    for i = 1:size(curSample.preds_deletion, 2)
        if numel(curSample.preds_deletion{2,i})<4
            boxes_pred_xywhc_i = [];
        else
            boxes_pred_xywhc_i = cat(2, curSample.preds_deletion{2,i}, curSample.preds_deletion{5,i}');   % Combine Box Position and Confidence
        end
        Res_i = calHR(num2cell(boxes_pred_xywhc_i,2), num2cell(boxes_gt_xywh,2), opt_vec);
        meanConf_deletionAI(i) = mean(Res_i.Confidence);

    end

end

function meanConf_insertationAI = getInsertionAI_res(curSample, boxes_gt_xywh, opt_vec)
    % Deletion: AI
    meanConf_insertationAI = zeros(size(curSample.preds_insertation, 2), 1);
    for i = 1:size(curSample.preds_insertation, 2)
        if numel(curSample.preds_insertation{2,i})<4
            boxes_pred_xywhc_i = [];
        else
            boxes_pred_xywhc_i = cat(2, curSample.preds_insertation{2,i}, curSample.preds_insertation{5,i}');   % Combine Box Position and Confidence
        end
        Res_i = calHR(num2cell(boxes_pred_xywhc_i,2), num2cell(boxes_gt_xywh,2), opt_vec);
        meanConf_insertationAI(i) = mean(Res_i.Confidence);

    end

end

function meanConf_deletionHuman = getDeletionHuman_res(curSample, boxes_gt_xywh, opt_vec)
    % Deletion: AI
    meanConf_deletionHuman = zeros(size(curSample.human_deletion, 2), 1);
    for i = 1:size(curSample.human_deletion, 2)
        if numel(curSample.human_deletion{2,i})<4
            boxes_pred_xywhc_i = [];
        else
            boxes_pred_xywhc_i = cat(2, curSample.human_deletion{2,i}, curSample.human_deletion{5,i}');   % Combine Box Position and Confidence
        end   
        Res_i = calHR(num2cell(boxes_pred_xywhc_i,2), num2cell(boxes_gt_xywh,2), opt_vec);
        meanConf_deletionHuman(i) = mean(Res_i.Confidence);

    end

end

function meanConf_insertationHuman = getInsertationHuman_res(curSample, boxes_gt_xywh, opt_vec)
    % Deletion: AI
    meanConf_insertationHuman = zeros(size(curSample.human_insertation, 2), 1);
    for i = 1:size(curSample.human_insertation, 2)
        if numel(curSample.human_insertation{2,i})<4
            boxes_pred_xywhc_i = [];
        else
            boxes_pred_xywhc_i = cat(2, curSample.human_insertation{2,i}, curSample.human_insertation{5,i}');   % Combine Box Position and Confidence
        end  
        Res_i = calHR(num2cell(boxes_pred_xywhc_i,2), num2cell(boxes_gt_xywh,2), opt_vec);
        meanConf_insertationHuman(i) = mean(Res_i.Confidence);

    end

end

% % function meanConf_deletionAI = getDeletionAI_res(curSample, boxes_gt_xywh, opt_vec)
% %     boxes_pred_xywhc_baseline = curSample.boxes_pred_xywhc_baseline;
% %     Res_baseline = calHR(num2cell(boxes_pred_xywhc_baseline,2), num2cell(boxes_gt_xywh,2), opt_vec);
% %     % Deletion: AI
% %     meanConf_deletionAI = zeros(size(curSample.preds_deletion, 2), 1);
% %     for i = 1:size(curSample.preds_deletion, 2)
% %         if numel(curSample.preds_deletion{2,i})<4
% %             boxes_pred_xywhc_i = [];
% %         else
% %             boxes_pred_xywhc_i = cat(2, curSample.preds_deletion{2,i}, curSample.preds_deletion{5,i}');   % Combine Box Position and Confidence
% %         end
% %         Res_i = calHR(num2cell(boxes_pred_xywhc_i,2), num2cell(boxes_gt_xywh,2), opt_vec);
% %         val = mean(max(0, Res_baseline.Confidence(Res_baseline.Confidence>0) - Res_i.Confidence(Res_baseline.Confidence>0))./Res_baseline.Confidence(Res_baseline.Confidence>0));
% %         
% %         meanConf_deletionAI(i) = mean(max(0, Res_baseline.Confidence(Res_baseline.Confidence>0) - Res_i.Confidence(Res_baseline.Confidence>0))./Res_baseline.Confidence(Res_baseline.Confidence>0));
% % 
% %     end
% % 
% % end
% % 
% % function meanConf_insertionAI = getInsertionAI_res(curSample, boxes_gt_xywh, opt_vec)
% %     boxes_pred_xywhc_baseline = curSample.boxes_pred_xywhc_baseline;
% %     Res_baseline = calHR(num2cell(boxes_pred_xywhc_baseline,2), num2cell(boxes_gt_xywh,2), opt_vec);
% %     % Insertion: AI
% %     meanConf_insertionAI = zeros(size(curSample.preds_insertation, 2), 1);
% % 
% %     for i = 1:size(curSample.preds_insertation, 2)
% %         if numel(curSample.preds_insertation{2,i})<4
% %             boxes_pred_xywhc_i = [];
% %         else
% %             boxes_pred_xywhc_i = cat(2, curSample.preds_insertation{2,i}, curSample.preds_insertation{5,i}');   % Combine Box Position and Confidence
% %         end
% %         Res_i = calHR(num2cell(boxes_pred_xywhc_i,2), num2cell(boxes_gt_xywh,2), opt_vec);
% %         meanConf_insertionAI(i) = mean(Res_baseline.Confidence < Res_i.Confidence);
% % 
% %     end
% % 
% % end
% % 
% % function meanConf_deletionHuman = getDeletionHuman_res(curSample, boxes_gt_xywh, opt_vec)
% %     boxes_pred_xywhc_baseline = curSample.boxes_pred_xywhc_baseline;
% %     Res_baseline = calHR(num2cell(boxes_pred_xywhc_baseline,2), num2cell(boxes_gt_xywh,2), opt_vec);
% %     % Deletion: AI
% %     meanConf_deletionHuman = zeros(size(curSample.human_deletion, 2), 1);
% %     for i = 1:size(curSample.human_deletion, 2)
% %         if numel(curSample.human_deletion{2,i})<4
% %             boxes_pred_xywhc_i = [];
% %         else
% %             boxes_pred_xywhc_i = cat(2, curSample.human_deletion{2,i}, curSample.human_deletion{5,i}');   % Combine Box Position and Confidence
% %         end   
% %         Res_i = calHR(num2cell(boxes_pred_xywhc_i,2), num2cell(boxes_gt_xywh,2), opt_vec);
% %         meanConf_deletionHuman(i) = mean(max(0, Res_baseline.Confidence(Res_baseline.Confidence>0) - Res_i.Confidence(Res_baseline.Confidence>0))./Res_baseline.Confidence(Res_baseline.Confidence>0));
% % 
% %     end
% % 
% % end
% % 
% % function meanConf_insertationHuman = getInsertationHuman_res(curSample, boxes_gt_xywh, opt_vec)
% %     boxes_pred_xywhc_baseline = curSample.boxes_pred_xywhc_baseline;
% %     Res_baseline = calHR(num2cell(boxes_pred_xywhc_baseline,2), num2cell(boxes_gt_xywh,2), opt_vec);
% %     % Deletion: AI
% %     meanConf_insertationHuman = zeros(size(curSample.human_insertation, 2), 1);
% %     for i = 1:size(curSample.human_insertation, 2)
% %         if numel(curSample.human_insertation{2,i})<4
% %             boxes_pred_xywhc_i = [];
% %         else
% %             boxes_pred_xywhc_i = cat(2, curSample.human_insertation{2,i}, curSample.human_insertation{5,i}');   % Combine Box Position and Confidence
% %         end  
% %         Res_i = calHR(num2cell(boxes_pred_xywhc_i,2), num2cell(boxes_gt_xywh,2), opt_vec);
% %         meanConf_insertationHuman(i) = mean(Res_baseline.Confidence < Res_i.Confidence);
% % 
% %     end
% % 
% % end

function Res = calHR(MouseList, TargetList, opt_vec)
    opt.BboxErr_Thr = opt_vec(1);
    opt.IoU_Thr = opt_vec(2);
    opt.ImageWidth = opt_vec(3);
    opt.ImageHeight = opt_vec(4);

    BboxErr_Thr = opt.BboxErr_Thr;
    IoU_Thr = opt.IoU_Thr;
    % Format: xywh
    varNames = {'GT_Bbox_Corr','Passive_Bbox_List','Hit_Count','Confidence'};
    varTypes = {'cell','cell','double','double'};
    sz = [numel(TargetList), numel(varTypes)];
    T_HR = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);

    for i = 1:numel(TargetList)
        curTarget = TargetList{i};
%         T_stats.TargetList{i} = curTarget;
        T_HR.GT_Bbox_Corr{i} = curTarget;
        MouseList_deleteIdx = [];
        for j = 1:numel(MouseList)
            curMouse = MouseList{j}; % Jinhan: why using top left corner? Why using this to determine if pred match target bb as their sizes are different?
            dist = distDot2Rect(curMouse(1), curMouse(2), curTarget(1), curTarget(2), curTarget(3), curTarget(4));
            if dist <= BboxErr_Thr
                T_HR.Passive_Bbox_List{i} = [T_HR.Passive_Bbox_List{i}; curMouse];
                T_HR.Hit_Count(i) = T_HR.Hit_Count(i) + 1;
                MouseList_deleteIdx = [MouseList_deleteIdx; j];
            end
        end
        MouseList(MouseList_deleteIdx) = [];
        
        if T_HR.Hit_Count(i) == 0
            for k = 1:(i-1)
                if size(T_HR.Passive_Bbox_List{k},1) > 1
                    passive_mouseMat = T_HR.Passive_Bbox_List{k};
                    rec_idx = [];
                    rec_dist = [];
                    for l = 1:size(passive_mouseMat,1)
                        curMouse = passive_mouseMat(l,:);
                        dist = distDot2Rect(curMouse(1), curMouse(2), curTarget(1), curTarget(2), curTarget(3), curTarget(4));
                        if dist <= BboxErr_Thr
                            rec_dist = [rec_dist; dist];
                            rec_idx = [rec_idx; l];
                        end
                    end
                    [minDist, minIdx] = min(rec_dist);
                    selIdx = rec_idx(minIdx);
                    T_HR.Passive_Bbox_List{i} = [T_HR.Passive_Bbox_List{i}; passive_mouseMat(selIdx,:)];
                    passive_mouseMat(selIdx,:) = [];
                    T_HR.Passive_Bbox_List{k} = passive_mouseMat;
                    T_HR.Hit_Count(k) = size(T_HR.Passive_Bbox_List{k},1);
                end
            end
            T_HR.Hit_Count(i) = size(T_HR.Passive_Bbox_List{i},1);
        end
    end
    HR_old = sum(T_HR.Hit_Count~=0)/numel(TargetList);  % The hit rate not considers IoU
    FalseAlarmList = cat(1,MouseList{:});
    
    for i = 1:numel(TargetList)
        curGT_Bbox_Corr = T_HR.GT_Bbox_Corr{i};
        if T_HR.Hit_Count(i) >= 1
            curPredictBboxList = T_HR.Passive_Bbox_List{i};
            IoU_rec = zeros(size(curPredictBboxList,1),1);
            for j = 1:size(curPredictBboxList,1)
                IoU_rec(j) = IoU_cal(curGT_Bbox_Corr, curPredictBboxList(j,1:4), opt);
            end
            [maxIoU, maxIoU_idx] = max(IoU_rec);
            hit_logicalVec = false(size(curPredictBboxList,1), 1);
            if maxIoU > IoU_Thr
                hit_logicalVec(maxIoU_idx) = true;
                hit_rec = curPredictBboxList(hit_logicalVec,:);
                falseAlarm_rec = curPredictBboxList(~hit_logicalVec,:);
            else
                hit_rec = [];
                falseAlarm_rec = curPredictBboxList;
            end
            T_HR.Passive_Bbox_List{i} = hit_rec;
            FalseAlarmList = cat(1, FalseAlarmList, falseAlarm_rec);

            T_HR.Hit_Count(i) = size(T_HR.Passive_Bbox_List{i},1);
        end
        
    end

    % Seperate Confidence
    for i = 1:numel(TargetList)
        Pred_Bbox_Corr = T_HR.Passive_Bbox_List{i};
        if ~isempty(Pred_Bbox_Corr)
            if size(Pred_Bbox_Corr,1)~=1
                error('This cell has multiple corrdinates');
            end
            T_HR.Confidence(i) = Pred_Bbox_Corr(end);
            T_HR.Passive_Bbox_List{i} = Pred_Bbox_Corr(1:end-1);
        else
            T_HR.Confidence(i) = 0; % prediction score
        end

    end

    HR = sum(T_HR.Hit_Count~=0)/numel(TargetList);  % The hit rate considers IoU

    Res.HitList = T_HR.Passive_Bbox_List(T_HR.Hit_Count~=0);
    Res.MissList = T_HR.GT_Bbox_Corr(T_HR.Hit_Count==0);
    Res.FalseAlarmList = num2cell(FalseAlarmList,2);
    Res.HR_withoutIoU = HR_old;
    Res.HR_withIoU = HR;
    Res.Confidence = T_HR.Confidence;

end

function IoU_rec = IoU_cal(curGT_Bbox_Corr, curPredictBbox, opt)
    H = opt.ImageHeight;
    W = opt.ImageWidth;
    M_GT = zeros(H, W);
    M_PR = zeros(H, W);
    GTBbox_xyxy = round(xywh2xyxy(curGT_Bbox_Corr))+1;
    PRBbox_xyxy = round(xywh2xyxy(curPredictBbox))+1;

    PRBbox_xyxy(3) = min(PRBbox_xyxy(3),W);
    PRBbox_xyxy(4) = min(PRBbox_xyxy(4),H);
    GTBbox_xyxy(3) = min(GTBbox_xyxy(3),W);
    GTBbox_xyxy(4) = min(GTBbox_xyxy(4),H);
    GTBbox_xyxy(GTBbox_xyxy<1) = 1;
    PRBbox_xyxy(PRBbox_xyxy<1) = 1;

    M_GT(GTBbox_xyxy(2):GTBbox_xyxy(4), GTBbox_xyxy(1):GTBbox_xyxy(3)) = 1;
    M_PR(PRBbox_xyxy(2):PRBbox_xyxy(4), PRBbox_xyxy(1):PRBbox_xyxy(3)) = 1;
    IoU_rec = sum(logical(M_GT.*M_PR),'all')./sum(logical(M_GT+M_PR),'all');

end

function dist = distDot2Rect(xMouse, yMouse, xloc, yloc, w, h)
    % Reference: https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
    minX = xloc - w / 2;
    maxX = xloc + w / 2;
    minY = yloc - h / 2;
    maxY = yloc + h / 2;
    px = xMouse;
    py = yMouse;

    dx = max([minX-px, 0, px-maxX]);
    dy = max([minY-py, 0, py-maxY]);

    dist = sqrt(dx.^2 + dy.^2);

end

function [MixedRawImg, PredRawImg, GTRawImg] = getMixedRawImg(curRawImg, Res)
    LineWidthVal = 1;
    rawImg = curRawImg;

    % Mixed
    curRawImg = rawImg;
    for i = 1:numel(Res.HitList)
        curCorr_xywh = Res.HitList{i};
        curCorr_xyxy = xywh2xyxy(curCorr_xywh);
        curCorr_xyMwh = [curCorr_xyxy(1:2) curCorr_xywh(3:4)];
        curRawImg = insertShape(curRawImg,"rectangle",curCorr_xyMwh,LineWidth=LineWidthVal,Color="green");

    end

    for i = 1:numel(Res.MissList)
        curCorr_xywh = Res.MissList{i};
        curCorr_xyxy = xywh2xyxy(curCorr_xywh);
        curCorr_xyMwh = [curCorr_xyxy(1:2) curCorr_xywh(3:4)];
        curRawImg = insertShape(curRawImg,"rectangle",curCorr_xyMwh,LineWidth=LineWidthVal,Color="blue");

    end

    for i = 1:numel(Res.FalseAlarmList)
        curCorr_xywh = Res.FalseAlarmList{i};
        curCorr_xyxy = xywh2xyxy(curCorr_xywh);
        curCorr_xyMwh = [curCorr_xyxy(1:2) curCorr_xywh(3:4)];
        curRawImg = insertShape(curRawImg,"rectangle",curCorr_xyMwh,LineWidth=LineWidthVal,Color="red");

    end

    MixedRawImg = curRawImg;

    % Pred
    curRawImg = rawImg;
    for i = 1:numel(Res.PredList)
        curCorr_xywh = Res.PredList{i};
        curCorr_xyxy = xywh2xyxy(curCorr_xywh);
        curCorr_xyMwh = [curCorr_xyxy(1:2) curCorr_xywh(3:4)];
        curRawImg = insertShape(curRawImg,"rectangle",curCorr_xyMwh,LineWidth=LineWidthVal,Color="green");

    end

    PredRawImg = curRawImg;

    % GT
    curRawImg = rawImg;
    for i = 1:numel(Res.GTList)
        curCorr_xywh = Res.GTList{i};
        curCorr_xyxy = xywh2xyxy(curCorr_xywh);
        curCorr_xyMwh = [curCorr_xyxy(1:2) curCorr_xywh(3:4)];
        curRawImg = insertShape(curRawImg,"rectangle",curCorr_xyMwh,LineWidth=LineWidthVal,Color="green");

    end

    GTRawImg = curRawImg;    

end


% function p_new = xyxy2xywh(p)
%     p_new = zeros(size(p),"like",p);
%     p_new(:,1) = (p(:,1)+p(:,3))./2;
%     p_new(:,2) = (p(:,2)+p(:,4))./2;
%     p_new(:,3) = p(:,3)-p(:,1);
%     p_new(:,4) = p(:,4)-p(:,2);
% 
% end
% 
% function p_new = xywh2xyxy(p)
%     p_new = zeros(size(p),"like",p);
%     p_new(:,1) = p(:,1)-p(:,3)./2;
%     p_new(:,2) = p(:,2)-p(:,4)./2;
%     p_new(:,3) = p(:,1)+p(:,3)./2;
%     p_new(:,4) = p(:,2)+p(:,4)./2;
% 
% end


