function Res = calHR(MouseList, TargetList, opt)
    BboxErr_Thr = opt.BboxErr_Thr;
    IoU_Thr = opt.IoU_Thr;
    % Format: xywh
    varNames = {'GT_Bbox_Corr','Passive_Bbox_List','Hit_Count'};
    varTypes = {'cell','cell','double'};
    sz = [numel(TargetList), numel(varTypes)];
    T_HR = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);

    for i = 1:numel(TargetList)
        curTarget = TargetList{i};
%         T_stats.TargetList{i} = curTarget;
        T_HR.GT_Bbox_Corr{i} = curTarget;
        MouseList_deleteIdx = [];
        for j = 1:numel(MouseList)
            curMouse = MouseList{j};
%             if numel(curMouse) < 4
%                 warning('No Valid Axis Info.');
%                 curMouse = [0 0 0 0];
%             end
            if numel(curMouse) < 2
                curMouse = zeros(2,1);
            end
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
                IoU_rec(j) = IoU_cal(curGT_Bbox_Corr, curPredictBboxList(j,:), opt);
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
    HR = sum(T_HR.Hit_Count~=0)/numel(TargetList);  % The hit rate considers IoU

    Res.HitList = T_HR.Passive_Bbox_List(T_HR.Hit_Count~=0);
    Res.MissList = T_HR.GT_Bbox_Corr(T_HR.Hit_Count==0);
    Res.FalseAlarmList = num2cell(FalseAlarmList,2);
    Res.HR_withoutIoU = HR_old;
    Res.HR_withIoU = HR;

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