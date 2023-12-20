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

% COCO PV
Path.RawDataPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\fixation\split_by_id\PV\';
Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\mscoco\attention_maps\231221_PV_raw\attention_maps\';
Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\mscoco\attention_maps\231221_PV_raw\visualize\';
Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\images\resized\DET\';

% COCO DET
Path.RawDataPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\fixation\split_by_id\DET\';
Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\mscoco\attention_maps\231221_DET_resized\attention_maps\';
Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\mscoco\attention_maps\231221_DET_resized\visualize\';
Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\images\resized\DET\';

% % COCO EXP
% Path.RawDataPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\fixation\split_by_id\PV\';
% Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\mscoco\attention_maps\231221_PV_raw\attention_maps\';
% Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\mscoco\attention_maps\231221_PV_raw\visualize\';
% Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\images\resized\DET\';


imginfoPath = 'H:\OneDrive - The University Of Hong Kong\mscoco\image_info\image_info_actualarea.xlsx';
imginfo = readtable(imginfoPath);

Path.CodePath = pwd;
addpath(genpath(Path.CodePath));

CheckIfDirExist(Path);

for i=1:height(imginfo)
    img_name = strrep(imginfo{i,'StimuliID'}{1},'.png','');

    intA = xlsread([Path.RawDataPath img_name '.xlsx']);

    img_path = [Path.backgroundPath img_name '.png'];

    % Realign fixations to the top left corner of the original, not resized
    % images. Read width and height data to limit the region on which
    % saliency maps are generated

    % Check if resized image is in different size than raw image
    raw_size = size(imread(img_path));
    % if ~((imginfo{i,'height'}==raw_size(1)) && (imginfo{i,'width'}==raw_size(2)))
    %     fprintf('%s\tRaw: %dx%d Area:%dx%d',img_name,raw_size(1),raw_size(2),imginfo{i,'height'},imginfo{i,'width'});
    %     continue
    % end

    intA(:,1) = intA(:,1) - imginfo{i,'Xlo'}; % FixX
    intA(:,2) = intA(:,2) - imginfo{i,'Ylo'}; % FixY

    opt.gaussian_smooth_val = 21;   
    [output_map] = gen_heatmap_single(intA(:,1:3), img_path,imginfo{i,'height'},imginfo{i,'width'},opt);
    output_map_norm = (output_map-min(output_map,[],'all'))./(max(output_map,[],'all')-min(output_map,[],'all'));
    
    % save mat file
    save(fullfile(Path.matSavePath, [img_name '_GSmo_' num2str(opt.gaussian_smooth_val) '.mat']), 'output_map_norm');

    % save plots
    copyfile('dataset1picedge.tiff', fullfile(Path.picSavePath, [img_name '_GSmo_' num2str(opt.gaussian_smooth_val) '.tiff']))

end


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
cfg.smoothingpic = opt.gaussian_smooth_val;
output_map = imap_gy(cfg);

end



