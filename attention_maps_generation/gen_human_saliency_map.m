% Generate human attention maps based for MSCOCO images, which have
% inconsistent image size, whereas human fixations cover the whole screen
%   - Jinhan Zhang 23-12-18

clear;

% % DET Hum
% Path.RawDataPath = 'H:\Projects\HKU_XAI_Project\XAI_Similarity_1\Hum_IdTask_Split_Fixation';
% Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\231206 Hum DET\whole_image';
% Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\231206 Hum DET\whole_image_visualize';
% Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\bdd\images\orib_hum_id_task_resized\';

% % EXP Hum
% Path.RawDataPath = 'H:\OneDrive - The University Of Hong Kong\bdd\results\explanation\231018_vehicle_whole_screen_vb_fixed_pos\exp_fixations_split_by_img';
% Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\231219 Veh EXP\attention_maps';
% Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\231219 Veh EXP\visualize';
% Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\bdd\images\orib_veh_id_task_resized\';

for c = 1:4
    if c==1
        % EXP Veh Grp1
        Path.RawDataPath = 'H:\OneDrive - The University Of Hong Kong\bdd\fixation\split_by_id\Veh_Yolo_ExpTask_Fixation_grp1';
        Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\240107 Veh EXP\grp1';
        Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\240107 Veh EXP\grp1_visualize';
        Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\bdd\images\orib_veh_id_task_resized\';
    end
    if c==2
        % EXP Veh Grp2
        Path.RawDataPath = 'H:\OneDrive - The University Of Hong Kong\bdd\fixation\split_by_id\Veh_Yolo_ExpTask_Fixation_grp2';
        Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\240107 Veh EXP\grp2';
        Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\240107 Veh EXP\grp2_visualize';
        Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\bdd\images\orib_veh_id_task_resized\';
    end
    if c==3
        % EXP Hum Grp1
        Path.RawDataPath = 'H:\OneDrive - The University Of Hong Kong\bdd\fixation\split_by_id\Hum_Yolo_ExpTask_Fixation_grp1';
        Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\240107 Hum EXP\grp1';
        Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\240107 Hum EXP\grp1_visualize';
        Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\bdd\images\orib_hum_id_task_resized\';
    end
    if c==4
        % EXP Hum Grp2
        Path.RawDataPath = 'H:\OneDrive - The University Of Hong Kong\bdd\fixation\split_by_id\Hum_Yolo_ExpTask_Fixation_grp2';
        Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\240107 Hum EXP\grp2';
        Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\240107 Hum EXP\grp2_visualize';
        Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\bdd\images\orib_hum_id_task_resized\';
    end

    Path.CodePath = pwd;
    addpath(genpath(Path.CodePath));
    
    CheckIfDirExist(Path);
    
    fileDir = dir(fullfile(Path.RawDataPath, ['*.xlsx']));

    for i = 1:numel(fileDir)

        img_name = strrep(fileDir(i).name, '.xlsx', '');
        img_name = strrep(img_name, '.jpg', '');
    
        curDir = fullfile(fileDir(i).folder, fileDir(i).name);
        
        intA = xlsread(curDir);

        img_path = [Path.backgroundPath img_name '.jpg']
    
        opt.gaussian_smooth_val = 30;   
        [output_map] = gen_heatmap_single(intA(:,1:3),img_path, 576,1024,opt);
        output_map_norm = (output_map-min(output_map,[],'all'))./(max(output_map,[],'all')-min(output_map,[],'all'));
        
        % save mat file
        save(fullfile(Path.matSavePath, [img_name '_GSmo_' num2str(opt.gaussian_smooth_val) '.mat']), 'output_map_norm');
    
        % save plots
        copyfile('dataset1picedge.tiff', fullfile(Path.picSavePath, [img_name '_GSmo_' num2str(opt.gaussian_smooth_val) '.tiff']))
    
    end

end





% % COCO PV
% Path.RawDataPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\fixation\split_by_id\PV\';
% Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\mscoco\attention_maps\231221_PV_resized\attention_maps\';
% Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\mscoco\attention_maps\231221_PV_resized\visualize\';
% Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\images\resized\DET\';
% 
% % COCO DET
% Path.RawDataPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\fixation\split_by_id\DET\';
% Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\mscoco\attention_maps\231221_DET_resized\attention_maps\';
% Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\mscoco\attention_maps\231221_DET_resized\visualize\';
% Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\images\resized\DET\';
% 
% % COCO EXP
% Path.RawDataPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\fixation\split_by_id\EXP_excluded_cleaned\';
% Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\mscoco\attention_maps\231222_EXP_excluded_cleaned_resized\attention_maps\';
% Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\mscoco\attention_maps\231222_EXP_excluded_cleaned_resized\visualize\';
% Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\images\resized\EXP\';


% imginfoPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\image_info\image_info_actualarea.xlsx';
% imginfo = readtable(imginfoPath);
% 
% Path.CodePath = pwd;
% addpath(genpath(Path.CodePath));
% 
% CheckIfDirExist(Path);
% 
% for i=1:height(imginfo)
%     img_name = strrep(imginfo{i,'StimuliID'}{1},'.png','');
% 
%     intA = xlsread([Path.RawDataPath img_name '.xlsx']);
% 
%     img_path = [Path.backgroundPath img_name '.png'];
% 
%     % Realign fixations to the top left corner of the original, not resized
%     % images. Read width and height data to limit the region on which
%     % saliency maps are generated
% 
%     % Check if resized image is in different size than raw image
%     raw_size = size(imread(img_path));
%     % if ~((imginfo{i,'height'}==raw_size(1)) && (imginfo{i,'width'}==raw_size(2)))
%     %     fprintf('%s\tRaw: %dx%d Area:%dx%d',img_name,raw_size(1),raw_size(2),imginfo{i,'height'},imginfo{i,'width'});
%     %     continue
%     % end
% 
%     intA(:,1) = intA(:,1) - imginfo{i,'Xlo'}; % FixX
%     intA(:,2) = intA(:,2) - imginfo{i,'Ylo'}; % FixY
% 
%     opt.gaussian_smooth_val = 21;   
%     [output_map] = gen_heatmap_single(intA(:,1:3), img_path,imginfo{i,'height'},imginfo{i,'width'},opt);
%     output_map_norm = (output_map-min(output_map,[],'all'))./(max(output_map,[],'all')-min(output_map,[],'all'));
% 
%     % save mat file
%     save(fullfile(Path.matSavePath, [img_name '_GSmo_' num2str(opt.gaussian_smooth_val) '.mat']), 'output_map_norm');
% 
%     % save plots
%     copyfile('dataset1picedge.tiff', fullfile(Path.picSavePath, [img_name '_GSmo_' num2str(opt.gaussian_smooth_val) '.tiff']))
% 
% end
% 
% 
%% Support Functions
function output_map = gen_heatmap_single(Sub, img_path, height, width, opt)

d1 = size(Sub,1);

summary = zeros(d1,4);
summary(:,1) = 1;
summary(:,2) = Sub(:,3);
summary(:,3) = Sub(:,1);
summary(:,4) = Sub(:,2);

save('data1.mat','summary');

cfg=[];
cfg.xSize= height; %height of the image, in pixel
cfg.ySize= width; %width of the image, in pixel
cfg.columnx=3;
cfg.columny=4;
cfg.columnduration=2;
cfg.columnitem=1;
cfg.dataset1= 1;
cfg.maptype = 1;
cfg.backgroundfile=img_path;    %'11.tiff'
cfg.smoothingpic = 30;
output_map = imap_gy(cfg);

end
% 
% 
% 
