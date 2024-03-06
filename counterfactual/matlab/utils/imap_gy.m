function output_map = imap_gy(cfg)

% imap toolbox for eye-trackin data analysis - Version 2.1
% http://perso.unifr.ch/roberto.caldara/index.php?page=4
% Sebastien Miellet & Roberto Caldara (2012). Junpeng Lao (2012) for the indtorgb function
% University of Fribourg
% sebastien.miellet@unifr.ch
%
% UPDATES DETAILS

% Version 2.1
% 1- Solve potential problems when the dimensions (x,y) of the search space is not an even number of pixels
% Version 2
% 1- The setting of the parameters is now done via a configuration structure (see examples), which allows more flexibility in calling iMap, but gives also to the user flexibility in inserting his own parameters.
% 2- New parameters have been added for: setting the colorbar scaling, setting the sigma (kernel for the statistical smoothing), setting the significancy of the threshold.
% 3- The "clicking step" used to generate the maps is no longer necessary. Many thanks to Junpeng Lao who wrote the 慽ndtorgb� function.
% 4- The one-tailed and two-tailed critical values (found in Zcrit.txt or displayed in the Matlab command window) are defined from the value of the significancy threshold set as one of the parameters.
% 5- The contours for significant areas are now displayed in white for all the fixation maps. It should improve the view of the significant areas.
% 6- A mistake in the data preparation code for the scenes example has been fixed.
% Version 1.1
% 1- Fixes potential problems with floating point in the calculation of the search-space size
% 2- Creates a Zcrit.txt file indicating the size of the search-space, the default critical value of a one-tailed Z for alpha = .05 (significancy threshold for the individual maps),
% the default critical value of a two-tailed Z for alpha = .05 (significancy threshold for the difference map). This information is also displayed in the Matlab command window.
% 3- The CiVol and STAT_THRESHOLD functions have been modified to avoid the display of confusing information in the Matlab command window.
%
% Disclaimer 
% iMap is a free software; you can redistribute it and/or modify it.
% We cannot be hold responsible for any damage that may (appear to) be caused by the use of iMap. Use at your own risk.
% This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
% Please cite us and let us know if you have any comment or suggestion.
% Thank you to the users who sent us feedbacks.
%
% VARIABLES that can be set in the cfg structure
% imap (xSize, ySize, columnx, columny, columnduration, columnitem, dataset1, dataset2, standard deviation, maptype, firstfix, backgroundfile, specificfix, searchspace), with
% 1-	xSize and ySize: stimulus size in pixels (e.g. 382, 390)
% IMPORTANT: Please keep in mind that the stimulis dimentions (xSize and ySize) might be inverted depending on whether the user considers them in graph coordinates (abscissa/ordinate, bottom left origin), screen coordinates (top left origin) or matrices (number of lines first, number of columns second). Here we consider matrix coordinates.
% 2-	columnx, columny, columnduration, columnitem: specify the column number for x, y coordinates, fixation durations and item number. This allows flexible data format. By defaults these columns are 1, 2, 3 and 4
% 3-	dataset1 and dataset2: specify the data .mat files that will be tested/compared. For example [1:20], [21:40] to compare data1 to data20 with data 21 to data40. The second data set is optional. If only one dataset is tested, iMap produces a statistical map and eye-tracking indexes for this dataset. If two datasets are specified, iMap provides the statistical maps and eye-tracking indexes for both dataset and the difference map and indexes.
% 4-	smoothingpic: Standard deviation in pixels of the Gaussian used for the data map smoothing. The default value is 10 pixels.
% 5-    smoothingstat: Standard deviation in pixels of the Gaussian used for the statistics smoothing (PIXEL-TEST). The default value is 10 pixels. FOR COHERENCE BETWEEN  THE DISPLAY AND THE STATISTICAL TEST, WE HIGHLY RECOMMAND TO USE A IDENTICAL VALUE FOR SMOOTHINGPIC AND SMOOTHINGSTAT IN CASE THE DEFAULT VALUES ARE NOT KEPT.
% 6-	maptype: 1 for fixation duration maps, 2 for number of fixations maps. The default value is 1.
% 7-	firstfix: This option allows to ignore the first fixation of each trial. This is particularly useful if the stimuli are centred and a central fixation cross is presented before the trials. 1 (default option) keeps all the fixations, 2 ignores the first fixation of each trial.
% 8-	backgroundfile: e.g. 'facebackground.tif'. This option allows adding a background picture to the statistical fixation maps. The value is optional and has to be set to 0 or [] in order to specify the subsequent variables.
% 9-	specificfix: To select one or several specific fixations. e.g. [3 3] or [1 3]. This value is optional.
% 10-	searchspace: By default the size of the stimulus, xSize * ySize.  The search space size can be specified by indicating directly the number of pixels it contains or by using a black and white picture (e.g. ?facemask.tif?) where the black mask indicates the search space.
% 11-   scaledownup: To be specified as a 2 values vector ([scaledown scaleup]). It allows to set the color coded scale. It does present the advantage to have the same scale for the individual (specific to datasets) maps and the difference map. We recommand to run iMap a first time without setting this parameter in order to get an idea of the range and then to set it in a second time
% 12-   sigthres: Significativity threshold for the Pixel-test (one-tailed for the individual maps, two-tailed for the difference map). By default 0.5 (0.025 for the difference map)
%
% HOW TO USE IT
% Set the parameters in a configuration strcuture (e.g. cfg.xSize=400).
% Default values will be used for non-specified parameters
% Call the imap2 function with the configuration structure (e.g. imap2(cfg)).
% See the example folders for more specific explanations.
%
% INPUT
% The input data is a matrix with a fixation per line. The only crucial data are the coordinates and duration of the fixations and the
%   item numbers. Any other column can be use for specifying your
%   conditions.
%
% OUTPUT
% imap creates .tif pictures of the single and difference fixation maps merged with a background picture. It displays the significant
%   areas (white contours) based on a Pixel-test. It also creates .tif pictures with normalized scales.
%   It creates .txt files with global eye-tracking measures for both datasets (called eyebasicdataset1.txt and eyebasicdataset2.txt).
%   The columns are: the number of fixations, the total fixation duration, the mean fixation duration, the path length and the mean
%   saccade length). The lines correspond to the raw data files (participants, sessions).
%   The imap toolbox creates a text file called Zscore.txt that include the mean Zscores in the significant area for (respective columns)
%   the dataset 1, dataset 2, dataset 1 in the area 1 and area 2 (areas in which the fixation durations are significantly longer for
%   dataset 1 and 2 respectively), dataset 2 in the area 1 and area 2.
%   It also creates a .txt file with the cohen's d between both datasets for area 1 and 2 (areas in which the fixation durations are
%   significantly longer for dataset 1 and 2 respectively). The file is called cohend.txt
%   Finally, imap creates .txt files with the eye-tracking data in both the significant areas and the rest of the picture. The files are
%   called eyeareadataset1.txt and eyeareadataset2.txt and are organised the following way: mean fixation duration for area 1 then for
%   area 2 then for the rest of the picture. In the same way are path length, total fixation duration and number of fixations.
%
% EXAMPLES 
% The data in the examples are in .mat files (called data1.mat, data2.mat,...), the matrices are called "summary". This can be obtained
%   from any txt file (e.g. fixation report from EyeLink, Data Viewer)
%
%
% CREDITS
% Exportfig was written by Ben Hinkle, 2001 (bhinkle@mathworks.com)
%
% CiVol and HalfMax are part of the Stat4Ci toolbox, which allows to perform the Pixel and the Cluster tests, both based on Random Field
%   Theory. These tests are easy to apply, requiring a mere four pieces of information; and they typically produce statistical thresholds
%   (or p-values) lower than the standard Bonferroni correction. An excellent non-technical reference is: K. J. Worsley (1996) the geometry
%   of random image. Chance, 9, 27-40.
%
% The STAT_THRESHOLD function was written by Keith Worsley for the fmristat toolbox (http://www.math.mcgill.ca/~keith/fmristat);
%   K.J. Worsley, 2003 (keith.worsley@mcgill.ca , www.math.mcgill.ca/keith)
%
% The Stat4Ci toolbox is free (http://www.mapageweb.umontreal.ca/gosselif/basic%20Stat4Ci%20tools/); if you use it in your research, please, cite :
%   Chauvin, A., Worsley, K. J., Schyns, P. G., Arguin, M. & Gosselin, F. (2004).  A sensitive statistical test for smooth classification images.
%   (http://journalofvision.org/5/9/1/)

%% Reading the cfg strucutre and setting default values
if isfield(cfg,'xSize')
    xSize=cfg.xSize;
else
    error('imap needs the size of the stimulus. Please specify cfg.xSize');
end
if isfield(cfg,'ySize')
    ySize=cfg.ySize;
else
    error('imap needs the size of the stimulus. Please specify cfg.ySize');
end
if isfield(cfg,'columnx')
    columnx=cfg.columnx;
else
    columnx=1;
end
if isfield(cfg,'columny')
    columny=cfg.columny;
else
    columny=2;
end
if isfield(cfg,'columnduration')
    columnduration=cfg.columnduration;
else
    columnduration=3;
end
if isfield(cfg,'columnitem')
    columnitem=cfg.columnitem;
else
    columnitem=4;
end
if isfield(cfg,'dataset1')
    dataset1=cfg.dataset1;
else
    error('imap needs at least 1 dataset. Please specify cfg.dataset1');
end
if isfield(cfg,'dataset2')
    numberofdataset=2;
    dataset2=cfg.dataset2;
else
    numberofdataset=1;
    dataset2=[];
end
if isfield(cfg,'specificfix')
    firstfix=cfg.specificfix(1);
else
    firstfix=1;
end
if isfield(cfg,'smoothingpic')
    smoothingpic=cfg.smoothingpic;
else
    smoothingpic=10;
end
if isfield(cfg,'smoothingstat')
    smoothingstat=cfg.smoothingstat;
else
    smoothingstat=10;
end
if isfield(cfg,'maptype')
    maptype=cfg.maptype;
else
    maptype=2;
end
if isfield(cfg,'firstfix')
    firstfix=cfg.firstfix;
else
    firstfix=1;
end
if isfield(cfg,'backgroundfile')
    backgroundfile=cfg.backgroundfile;
else
    backgroundfile=[];
end

if isfield(cfg,'searchspace')
    if ischar(searchspace);
        mask = double(imread(searchspace));
        pixsearchspace = (mask - min(mask(:))) / (max(mask(:)) - min(mask(:)));
        pixsearchspace=sum(pixsearchspace(:));
    else
        pixsearchspace=searchspace;
    end
else
    pixsearchspace = ones(xSize, ySize, 1);
    pixsearchspace=sum(pixsearchspace(:));
end

%% Global eye-tracking measures for each dataset
datatotal=[dataset1 dataset2];
for datasetnb=1:numberofdataset
    if datasetnb==1
        dataset=dataset1;
        datasetname='dataset1';
    elseif datasetnb==2
        dataset=dataset2;
        datasetname='dataset2';
    end
    
    nbfixtrial=[];
    totalfixdur=[];
    meanfixdur=[];
    pathlength=[];
    meansacclength=[];
    
    for datafilenumber=1:length(dataset)
        summary=[];
        datatoload=['data' num2str(dataset(datafilenumber))];
        load(datatoload); % The name of the matrix is 'summary'
        [nbfix nbvariables]=size(summary);
        cumulfixdur=[];
        cumulsaccadelength=[];
        numfix=[];
        cumulfixdur(1)=summary(1,columnduration);
        cumulsaccadelength(1)=0;
        numfix(1)=1;
        trialnb=0;
        for fix=2:nbfix
            if summary(fix,columnitem)==summary(fix-1,columnitem)
                numfix(fix)=numfix(fix-1)+1;
                cumulfixdur(fix)=cumulfixdur(fix-1)+summary(fix,columnduration);
                cumulsaccadelength(fix)=cumulsaccadelength(fix-1)+sqrt((summary(fix,columnx)-summary(fix-1,columnx)).^2+(summary(fix,columny)-summary(fix-1,columny)).^2);
            elseif summary(fix,columnitem)~=summary(fix-1,columnitem)
                trialnb=trialnb+1;
                nbfixtrial(trialnb,datafilenumber)=numfix(fix-1);
                totalfixdur(trialnb,datafilenumber)=cumulfixdur(fix-1);
                meanfixdur(trialnb,datafilenumber)=cumulfixdur(fix-1)/numfix(fix-1);
                pathlength(trialnb,datafilenumber)=cumulsaccadelength(fix-1);
                meansacclength(trialnb,datafilenumber)=cumulsaccadelength(fix-1)/numfix(fix-1);
                cumulfixdur(fix)=summary(fix,columnduration);
                cumulsaccadelength(fix)=0;
                numfix(fix)=1;
            end
        end
        trialnb=trialnb+1;
        nbfixtrial(trialnb,datafilenumber)=numfix(fix);
        nbfixtrial(nbfixtrial==0)=NaN;
        totalfixdur(trialnb,datafilenumber)=cumulfixdur(fix);
        totalfixdur(totalfixdur==0)=NaN;
        meanfixdur(trialnb,datafilenumber)=cumulfixdur(fix)/numfix(fix);
        meanfixdur(meanfixdur==0)=NaN;
        pathlength(trialnb,datafilenumber)=cumulsaccadelength(fix);
        pathlength(pathlength==0)=NaN;
        meansacclength(trialnb,datafilenumber)=cumulsaccadelength(fix)/numfix(fix);
        meansacclength(meansacclength==0)=NaN;
    end
    
    nbfixsbj=(nanmean(nbfixtrial))';
    totalfixdursbj=(nanmean(totalfixdur))';
    meanfixdursbj=(nanmean(meanfixdur))';
    pathlengthsbj=(nanmean(pathlength))';
    meansacclengthsbj=(nanmean(meansacclength))';
    
    if datasetnb==1
        eyebasicdataset1=[nbfixsbj totalfixdursbj meanfixdursbj pathlengthsbj meansacclengthsbj];
        dlmwrite('eyebasicdataset1.txt', eyebasicdataset1, 'delimiter', '\t', 'precision', 7);
        nbfixsbj=[];
        totalfixdursbj=[];
        meanfixdursbj=[];
        pathlengthsbj=[];
        meansacclengthsbj=[];
    elseif datasetnb==2
        eyebasicdataset2=[nbfixsbj totalfixdursbj meanfixdursbj pathlengthsbj meansacclengthsbj];
        dlmwrite('eyebasicdataset2.txt', eyebasicdataset2, 'delimiter', '\t', 'precision', 7);
        nbfixsbj=[];
        totalfixdursbj=[];
        meanfixdursbj=[];
        pathlengthsbj=[];
        meansacclengthsbj=[];
    end
end

%% Descriptive fixation maps
for datafilenumber=1:length(datatotal)
    summary=[];
    
    % POINTS step: Creating a matrix x by y (stimulus size) with the cumulated fixation
    % durations for each pixel
    datatoload=['data' num2str(datatotal(datafilenumber))];
    load(datatoload); % The name of the matrix is 'summary'
    [nbfix nbvariables]=size(summary);
    if isfield(cfg,'specificfix')
        nbfix=cfg.specificfix(2);
    end
    
    matrixduration = zeros(xSize, ySize);
    for fix=firstfix:nbfix
        if firstfix>=2
            if summary(fix, columnitem)==summary(fix-1, columnitem)
                coordX = round(summary(fix, columny)); % here we swap x and y (difference between screen coordinates and matrix [first number=lines, second=columns])
                coordY = round(summary(fix, columnx));
                if coordX<xSize && coordY<ySize && coordX>0 && coordY>0 % In this example, we consider only the fixations on the stimulus
                    if maptype==1
                        matrixduration(coordX, coordY) = matrixduration(coordX, coordY) + summary(fix, columnduration);
                    elseif maptype==2
                        matrixduration(coordX, coordY) = matrixduration(coordX, coordY) + 1;
                    end
                end
            end
        elseif firstfix==1
            coordX = round(summary(fix, columny)); % here we swap x and y (difference between screen coordinates and matrix [first number=lines, second=columns])
            coordY = round(summary(fix, columnx));
            if coordX<xSize && coordY<ySize && coordX>0 && coordY>0 % In this example, we consider only the fixations on the stimulus
                if maptype==1
                    matrixduration(coordX, coordY) = matrixduration(coordX, coordY) + summary(fix, columnduration);
                elseif maptype==2
                    matrixduration(coordX, coordY) = matrixduration(coordX, coordY) + 1;
                end
            end
        end
    end
    
    % SMOOTHING step
    if mod(xSize,2)==0 && mod(ySize,2)==0
        [x, y] = meshgrid(-floor(ySize/2)+.5:floor(ySize/2)-.5, -floor(xSize/2)+.5:floor(xSize/2)-.5);
    elseif mod(xSize,2)==1 && mod(ySize,2)==0
        [x, y] = meshgrid(-floor(ySize/2)+.5:floor(ySize/2)-.5, -floor(xSize/2)+.5:floor(xSize/2)-.5+1);
    elseif mod(xSize,2)==0 && mod(ySize,2)==1
        [x, y] = meshgrid(-floor(ySize/2)+.5:floor(ySize/2)-.5+1, -floor(xSize/2)+.5:floor(xSize/2)-.5);
    elseif mod(xSize,2)==1 && mod(ySize,2)==1
        [x, y] = meshgrid(-floor(ySize/2)+.5:floor(ySize/2)-.5+1, -floor(xSize/2)+.5:floor(xSize/2)-.5+1);
    end
    
    gaussienne = exp(- (x .^2 / smoothingpic ^2) - (y .^2 / smoothingpic ^2));
    gaussienne = (gaussienne - min(gaussienne(:))) / (max(gaussienne(:)) - min(gaussienne(:)));
    f_fil = fft2(gaussienne);
    f_mat = fft2(matrixduration); % 2D fourrier transform on the points matrix
    filtered_mat = f_mat .* f_fil;
    smoothpic = real(fftshift(ifft2(filtered_mat))); % take the real part of the complex values from the fourier transform
    nametosave=['data' num2str(datatotal(datafilenumber)) 'smoothpic'];
    save(nametosave, 'smoothpic');
end
clear filtered_mat f_mat f_fil x y smoothpic gaussienne;

%% Statistics
% Pool the maps for each dataset
for datasetnb=1:numberofdataset
    if datasetnb==1
        dataset=dataset1;
    elseif datasetnb==2
        dataset=dataset2;
    end
    pooldata = zeros(xSize, ySize);
    for ii=1:length(dataset)
        nametoload=['data' int2str(dataset(ii)) 'smoothpic'];
        load(nametoload);
        pooldata=smoothpic+pooldata;
    end
    
    % Normalizing the data
    theMean = mean(pooldata(:));
    stdev = std(pooldata(:));
    Zsmooth=(pooldata - theMean)/stdev;
    
    if datasetnb==1
        dataset1map=Zsmooth;
    elseif datasetnb==2
        dataset2map=Zsmooth;
    end
end

% If there are 2 datasets
% Compute the difference matrices, between the groups in this example but
% it can be done between conditions as well
if isempty(dataset2)==0
    diffmap=dataset1map-dataset2map;
    theMean = mean(diffmap(:));
    stdev = std(diffmap(:));
    Zdiffmap=(diffmap - theMean)/stdev;
end

%*****************
% Pixel test
%*****************
% for details see Chauvin, A., Worsley, K. J., Schyns, P. G., Arguin, M. &
% Gosselin, F. (2004).  A sensitive statistical test for smooth
% classification images.

% calculation of the significant values for single matrix
format short g
if isfield(cfg,'sigthres')
    p=cfg.sigthres;
else
    p=.05;	% desired p-value (Voxel test)
end
FWHM=HalfMax(smoothingstat);% computes the full width half maximum
[volumes,N]=CiVol(pixsearchspace,2); % (Worsley et al. 1996, HBM) computes the intrinsic volumes
[tP1,k]=stat_threshold(volumes, N,FWHM,Inf,p,0.05);
disp('search-space size (number of pixels)');
disp(pixsearchspace);
disp('threshold for individual maps (Zcrit, one-tailed test)');
disp(tP1);
% for the differences
if isfield(cfg,'sigthres')
    p=cfg.sigthres/2;
else
    p=.025;	% desired p-value (Voxel test)
end
[tP2,k]=stat_threshold(volumes, N,FWHM,Inf,p,0.05);
disp('threshold for difference maps (Zcrit, two-tailed test)');
disp(tP2);
Zcrit=[pixsearchspace tP1 tP2];
dlmwrite('Zcrit.txt', Zcrit, 'delimiter', '\t', 'precision', 7);
% Find the significant area in the single maps
Seuildataset1Sup = gt(dataset1map, tP1);
Seuildataset1Inf = lt(dataset1map, -tP1);
mask_dataset1=dataset1map.*Seuildataset1Sup;
Zscore_dataset1=sum(sum(mask_dataset1))/length(find(mask_dataset1~=0));
SeuilEdgeDataset1Sup = edge(double(Seuildataset1Sup),'canny',[0.01 0.11], 2);
SeuilEdgeDataset1Inf = edge(double(Seuildataset1Inf),'canny',[0.01 0.11], 2);
% If there are 2 datasets
if isempty(dataset2)==0
    Seuildataset2Sup = gt(dataset2map, tP1);
    Seuildataset2Inf = lt(dataset2map, -tP1);
    mask_dataset2=dataset2map.*Seuildataset2Sup;
    Zscore_dataset2=sum(sum(mask_dataset2))/length(find(mask_dataset2~=0));
    SeuilEdgeDataset2Sup = edge(double(Seuildataset2Sup),'canny',[0.01 0.11], 2);
    SeuilEdgeDataset2Inf = edge(double(Seuildataset2Inf),'canny',[0.01 0.11], 2);
    
    % Find the significant area in the difference map
    SeuilZDiffmapSup = gt(Zdiffmap, tP2);
    SeuilZDiffmapInf = lt(Zdiffmap, -tP2);
    mask_dataset1_area1=dataset1map.*SeuilZDiffmapSup;
    mask_dataset1_area2=dataset1map.*SeuilZDiffmapInf;
    mask_dataset2_area1=dataset2map.*SeuilZDiffmapSup;
    mask_dataset2_area2=dataset2map.*SeuilZDiffmapInf;
    Zscore_dataset1_area1=sum(sum(mask_dataset1_area1))/length(find(mask_dataset1_area1~=0));
    Zscore_dataset1_area2=sum(sum(mask_dataset1_area2))/length(find(mask_dataset1_area2~=0));
    Zscore_dataset2_area1=sum(sum(mask_dataset2_area1))/length(find(mask_dataset2_area1~=0));
    Zscore_dataset2_area2=sum(sum(mask_dataset2_area2))/length(find(mask_dataset2_area2~=0));
    SeuilEdgeDiffSup = edge(double(SeuilZDiffmapSup),'canny',[0.01 0.11], 2);
    SeuilEdgeDiffInf = edge(double(SeuilZDiffmapInf),'canny',[0.01 0.11], 2);
    for ll=1:2
        countstdZ=0;
        for kk=1:2
            if kk==1
                poolstdZ=dataset1map;
            elseif kk==2
                poolstdZ=dataset2map;
            end
            for ii=1:xSize
                for jj=1:ySize
                    if ll==1
                        if SeuilZDiffmapSup(ii,jj)~=0
                            countstdZ=countstdZ+1;
                            tempstdZ_area1(countstdZ)=poolstdZ(ii,jj);
                        end
                    elseif ll==2
                        if SeuilZDiffmapInf(ii,jj)~=0
                            countstdZ=countstdZ+1;
                            tempstdZ_area2(countstdZ)=poolstdZ(ii,jj);
                        end
                    end
                end
            end
        end
    end
    d_area1=(Zscore_dataset1_area1-Zscore_dataset2_area1)/std(tempstdZ_area1);
    d_area2=(Zscore_dataset1_area2-Zscore_dataset2_area2)/std(tempstdZ_area2);
    Zscore=[Zscore_dataset1 Zscore_dataset2 Zscore_dataset1_area1 Zscore_dataset1_area2 Zscore_dataset2_area1 Zscore_dataset2_area2];
    dlmwrite('Zscore.txt', Zscore, 'delimiter', '\t', 'precision', 7);
    cohend=[d_area1 d_area2];
    dlmwrite('cohend.txt', cohend, 'delimiter', '\t', 'precision', 7);
end

%% make figure with background (is specified) and significant areas
% export the single and difference maps in tiff format
if isfield(cfg,'scaledownup')
    scaledown=cfg.scaledownup(1);
    scaleup=cfg.scaledownup(2);
    figure, imagesc(dataset1map,[scaledown scaleup]), colorbar
else
    scaledown=[];
    scaleup=[];
    figure, imagesc(dataset1map), colorbar
end
output_map = dataset1map;
exportfig(gcf, 'dataset1map', 'Format', 'tiff', 'FontMode','fixed','FontSize', 10, 'color', 'cmyk' );
close(gcf);

if isempty(dataset2)==0
    if isfield(cfg,'scaledownup')
        scaledown=cfg.scaledownup(1);
        scaleup=cfg.scaledownup(2);
        figure, imagesc(dataset2map,[scaledown scaleup]), colorbar
    else
        scaledown=[];
        scaleup=[];
        figure, imagesc(dataset2map), colorbar
    end
    exportfig(gcf, 'dataset2map', 'Format', 'tiff', 'FontMode','fixed','FontSize', 10, 'color', 'cmyk' );
    close(gcf);
    if isfield(cfg,'scaledownup')
        scaledown=cfg.scaledownup(1);
        scaleup=cfg.scaledownup(2);
        figure, imagesc(Zdiffmap,[scaledown scaleup]), colorbar
    else
        scaledown=[];
        scaleup=[];
        figure, imagesc(Zdiffmap), colorbar
    end
    exportfig(gcf, 'Zdiffmap', 'Format', 'tiff', 'FontMode','fixed','FontSize', 10, 'color', 'cmyk' );
    close(gcf);
end

dataset1cmat_temp=colormap;
dataset1pic=indtorgb(dataset1map,scaledown,scaleup,dataset1cmat_temp);

if isempty(dataset2)==0
    dataset2cmat_temp=colormap;    
    dataset2pic=indtorgb(dataset2map,scaledown,scaleup,dataset2cmat_temp);

    Zdiffcmat_temp=colormap;   
    diffpic=indtorgb(Zdiffmap,scaledown,scaleup,Zdiffcmat_temp);
end
close all;

% add background if a background picture is specified
if isfield(cfg,'backgroundfile')
    % open the background picture
    imbackground = double(imread(sprintf(backgroundfile)))/255;
    % add maps to background
    if ndims(imbackground)==2
        im3D = zeros(xSize, ySize, 3);
        im3D(:, :, 1) = imbackground;
        im3D(:, :, 2) = imbackground;
        im3D(:, :, 3) = imbackground;
    elseif ndims(imbackground)==3
        im3D = zeros(xSize, ySize, 3);
        im3D(:, :, 1) = imbackground(:, :, 1);
        im3D(:, :, 2) = imbackground(:, :, 2);
        im3D(:, :, 3) = imbackground(:, :, 3);
    end
    dataset1pic2 = im3D * .3 + dataset1pic * .7;
    if isempty(dataset2)==0
        diffpic2 = im3D * .3 + diffpic * .7;
        dataset2pic2 = im3D * .3 + dataset2pic * .7;
    end
    
else
    
    dataset1pic2 = dataset1pic;
    if isempty(dataset2)==0
        dataset2pic2 = dataset2pic;
        diffpic2 = diffpic;
    end
end

% add contour for the significant areas and save
imSeuilEdge3D = zeros(xSize, ySize, 3);
imSeuilEdge3D(:, :, 1) = SeuilEdgeDataset1Sup;
imSeuilEdge3D(:, :, 2) = SeuilEdgeDataset1Sup;
imSeuilEdge3D(:, :, 3) = SeuilEdgeDataset1Sup;
dataset1picedge= (dataset1pic2).*255;
name = sprintf('dataset1picedge.tiff');
imwrite(uint8(dataset1picedge), name)
if isempty(dataset2)==0
    imSeuilEdge3D = zeros(xSize, ySize, 3);
    imSeuilEdge3D(:, :, 1) = SeuilEdgeDiffSup;
    imSeuilEdge3D(:, :, 2) = SeuilEdgeDiffSup;
    imSeuilEdge3D(:, :, 3) = SeuilEdgeDiffSup;
    imSeuilEdge3D2 = zeros(xSize, ySize, 3);
    imSeuilEdge3D2(:, :, 1) = SeuilEdgeDiffInf;
    imSeuilEdge3D2(:, :, 2) = SeuilEdgeDiffInf;
    imSeuilEdge3D2(:, :, 3) = SeuilEdgeDiffInf;
    diffpicedge= (diffpic2 + imSeuilEdge3D + imSeuilEdge3D2).*255;
    name = sprintf('diffpicedge.tiff');
    imwrite(uint8(diffpicedge), name)
    imSeuilEdge3D = zeros(xSize, ySize, 3);
    imSeuilEdge3D(:, :, 1) = SeuilEdgeDataset2Sup;
    imSeuilEdge3D(:, :, 2) = SeuilEdgeDataset2Sup;
    imSeuilEdge3D(:, :, 3) = SeuilEdgeDataset2Sup;
    dataset2picedge= (dataset2pic2).*255;
    name = sprintf('dataset2picedge.tiff');
    imwrite(uint8(dataset2picedge), name)
end

%% Effect sizes & eye-tracking measures in the significant areas
% Compute the cohen's d (using pooled variance) for each of the areas significantly different between
% datasets (according to the difference matrix)

if isempty(dataset2)==0
    % number of significant pixels in single and difference matrices
    nbpixeltotal=xSize*ySize;
    nbsignifpixelarea1=sum(sum(SeuilZDiffmapSup));
    nbsignifpixelarea2=sum(sum(SeuilZDiffmapInf));
    nbpixelother=nbpixeltotal-nbsignifpixelarea1-nbsignifpixelarea2;
    
    for datafilenumber=1:length(datatotal)
        summary=[];
        durationarea1=[];
        durationarea2=[];
        meanfixdurarea1=[];
        meanfixdurarea2=[];
        meanfixdurrest=[];
        pathlengtharea1=[];
        pathlengtharea2=[];
        pathlenthrest=[];
        totfixdurarea1=[];
        totfixdurarea2=[];
        totfixdurrest=[];
        numfixarea1=[];
        numfixarea2=[];
        numfixrest=[];
        
        datatoload=['data' num2str(datatotal(datafilenumber))];
        load(datatoload); % The name of the matrix is 'summary'
        [nbfix nbvariables]=size(summary);
        
        nbitem=0;
        nbfixarea1=0;
        nbfixarea2=0;
        nbfixrest=0;
        cumuldurationarea1=0;
        cumuldurationarea2=0;
        cumuldurationrest=0;
        cumulsaccadearea1=0;
        cumulsaccadearea2=0;
        cumulsaccaderest=0;
        coordX = round(summary(1, columny));
        coordY = round(summary(1, columnx));
        if coordX<xSize && coordY<ySize && coordX>0 && coordY>0
            if SeuilZDiffmapSup(coordX(1),coordY(1))~=0
                cumuldurationarea1= cumuldurationarea1+ summary(1, columnduration);
                nbfixarea1=nbfixarea1+1;
            elseif SeuilZDiffmapInf(coordX(1),coordY(1))~=0
                cumuldurationarea2= cumuldurationarea2+ summary(1, columnduration);
                nbfixarea2=nbfixarea2+1;
            else
                cumuldurationrest=cumuldurationrest+summary(1, columnduration);
                nbfixrest=nbfixrest+1;
            end
        end
        for fix=2:nbfix
            coordX = round(summary(fix, columny)); % here we swap x and y (difference between screen coordinates and matrix [first number=lines, second=columns])
            coordY = round(summary(fix, columnx));
            
            if coordX>0 && coordY>0 && coordX<xSize && coordY<ySize
                if summary(fix,columnitem)==summary(fix-1,columnitem)
                    if SeuilZDiffmapSup(coordX,coordY)~=0
                        cumuldurationarea1= cumuldurationarea1+ summary(fix, columnduration);
                        nbfixarea1=nbfixarea1+1;
                        cumulsaccadearea1=cumulsaccadearea1+sqrt((summary(fix,columnx)-summary(fix-1,columnx)).^2+(summary(fix,columny)-summary(fix-1,columny)).^2);
                    elseif SeuilZDiffmapInf(coordX,coordY)~=0
                        cumuldurationarea2= cumuldurationarea2+ summary(fix, columnduration);
                        nbfixarea2=nbfixarea2+1;
                        cumulsaccadearea2=cumulsaccadearea2+sqrt((summary(fix,columnx)-summary(fix-1,columnx)).^2+(summary(fix,columny)-summary(fix-1,columny)).^2);
                    else
                        cumuldurationrest=cumuldurationrest+summary(fix, columnduration);
                        nbfixrest=nbfixrest+1;
                        cumulsaccaderest=cumulsaccaderest+sqrt((summary(fix,columnx)-summary(fix-1,columnx)).^2+(summary(fix,columny)-summary(fix-1,columny)).^2);
                    end
                elseif summary(fix,columnitem)~=summary(fix-1,columnitem)
                    nbitem=nbitem+1;
                    meanfixdurarea1(nbitem)=cumuldurationarea1/nbfixarea1;
                    meanfixdurarea2(nbitem)=cumuldurationarea2/nbfixarea2;
                    meanfixdurrest(nbitem)=cumuldurationrest/nbfixrest;
                    pathlengtharea1(nbitem)=cumulsaccadearea1;
                    pathlengtharea2(nbitem)=cumulsaccadearea2;
                    pathlenthrest(nbitem)=cumulsaccaderest;
                    totfixdurarea1(nbitem)=cumuldurationarea1;
                    totfixdurarea2(nbitem)=cumuldurationarea2;
                    totfixdurrest(nbitem)=cumuldurationrest;
                    numfixarea1(nbitem)=nbfixarea1;
                    numfixarea2(nbitem)=nbfixarea2;
                    numfixrest(nbitem)=nbfixrest;
                    
                    nbfixarea1=0;
                    nbfixarea2=0;
                    nbfixrest=0;
                    cumuldurationarea1=0;
                    cumuldurationarea2=0;
                    cumuldurationrest=0;
                    cumulsaccadearea1=0;
                    cumulsaccadearea2=0;
                    cumulsaccaderest=0;
                    coordX = round(summary(1, columny));
                    coordY = round(summary(1, columnx));
                    if coordX>0 && coordY>0 && coordX<xSize && coordY<ySize
                        if SeuilZDiffmapSup(coordX,coordY)~=0
                            cumuldurationarea1= cumuldurationarea1+ summary(fix, columnduration);
                            nbfixarea1=nbfixarea1+1;
                        elseif SeuilZDiffmapInf(coordX,coordY)~=0
                            cumuldurationarea2= cumuldurationarea2+ summary(fix, columnduration);
                            nbfixarea2=nbfixarea2+1;
                        else
                            cumuldurationrest=cumuldurationrest+summary(fix, columnduration);
                            nbfixrest=nbfixrest+1;
                        end
                    end
                end
            end
        end
        
        coordX = round(summary(fix, columny));
        coordY = round(summary(fix, columnx));
        if coordX>0 && coordY>0 && coordX<xSize && coordY<ySize
            if SeuilZDiffmapSup(coordX,coordY)~=0
                cumuldurationarea1= cumuldurationarea1+ summary(fix, columnduration);
                nbfixarea1=nbfixarea1+1;
                cumulsaccadearea1=cumulsaccadearea1+sqrt((summary(fix,columnx)-summary(fix-1,columnx)).^2+(summary(fix,columny)-summary(fix-1,columny)).^2);
            elseif SeuilZDiffmapInf(coordX,coordY)~=0
                cumuldurationarea2= cumuldurationarea2+ summary(fix, columnduration);
                nbfixarea2=nbfixarea2+1;
                cumulsaccadearea2=cumulsaccadearea2+sqrt((summary(fix,columnx)-summary(fix-1,columnx)).^2+(summary(fix,columny)-summary(fix-1,columny)).^2);
            else
                cumuldurationrest=cumuldurationrest+summary(fix, columnduration);
                nbfixrest=nbfixrest+1;
                cumulsaccaderest=cumulsaccaderest+sqrt((summary(fix,columnx)-summary(fix-1,columnx)).^2+(summary(fix,columny)-summary(fix-1,columny)).^2);
            end
        end
        
        nbitem=nbitem+1;
        meanfixdurarea1(nbitem)=cumuldurationarea1/nbfixarea1;
        meanfixdurarea2(nbitem)=cumuldurationarea2/nbfixarea2;
        meanfixdurrest(nbitem)=cumuldurationrest/nbfixrest;
        pathlengtharea1(nbitem)=cumulsaccadearea1;
        pathlengtharea2(nbitem)=cumulsaccadearea2;
        pathlengthrest(nbitem)=cumulsaccaderest;
        totfixdurarea1(nbitem)=cumuldurationarea1;
        totfixdurarea2(nbitem)=cumuldurationarea2;
        totfixdurrest(nbitem)=cumuldurationrest;
        numfixarea1(nbitem)=nbfixarea1;
        numfixarea2(nbitem)=nbfixarea2;
        numfixrest(nbitem)=nbfixrest;
        
        mediandurationarea1(datafilenumber)=nanmedian(meanfixdurarea1);
        mediandurationarea2(datafilenumber)=nanmedian(meanfixdurarea2);
        mediandurationrest(datafilenumber)=nanmedian(meanfixdurrest);
        totalduration(datafilenumber)=mediandurationarea1(datafilenumber)+mediandurationarea2(datafilenumber)+mediandurationrest(datafilenumber);
        relativedurationarea1(datafilenumber)=(mediandurationarea1(datafilenumber)/nbsignifpixelarea1)/(totalduration(datafilenumber)/nbpixeltotal);
        relativedurationarea2(datafilenumber)=(mediandurationarea2(datafilenumber)/nbsignifpixelarea2)/(totalduration(datafilenumber)/nbpixeltotal);
        relativedurationrest(datafilenumber)=(mediandurationrest(datafilenumber)/nbpixelother)/(totalduration(datafilenumber)/nbpixeltotal);
        
        fixdurarea1sbj(datafilenumber)=nanmean(meanfixdurarea1');
        fixdurarea2sbj(datafilenumber)=nanmean(meanfixdurarea2');
        fixdurrestsbj(datafilenumber)=nanmean(meanfixdurrest');
        pathlengtharea1sbj(datafilenumber)=nanmean(pathlengtharea1');
        pathlengtharea2sbj(datafilenumber)=nanmean(pathlengtharea2');
        pathlengthrestsbj(datafilenumber)=nanmean(pathlengthrest');
        totfixdurarea1sbj(datafilenumber)=nanmean(totfixdurarea1');
        totfixdurarea2sbj(datafilenumber)=nanmean(totfixdurarea2');
        totfixdurrestsbj(datafilenumber)=nanmean(totfixdurrest');
        numfixarea1sbj(datafilenumber)=nanmean(numfixarea1');
        numfixarea2sbj(datafilenumber)=nanmean(numfixarea2');
        numfixrestsbj(datafilenumber)=nanmean(numfixrest');
        
    end
    
    nbdatapointdataset1=1;
    nbdatapointdataset2=1;
    for ii=1:length(datatotal)
        if ismember(datatotal(ii),dataset1)
            fixdurarea1dataset1(nbdatapointdataset1)=fixdurarea1sbj(ii);
            fixdurarea2dataset1(nbdatapointdataset1)=fixdurarea2sbj(ii);
            fixdurrestdataset1(nbdatapointdataset1)=fixdurrestsbj(ii);
            pathlengtharea1dataset1(nbdatapointdataset1)=pathlengtharea1sbj(ii);
            pathlengtharea2dataset1(nbdatapointdataset1)=pathlengtharea2sbj(ii);
            pathlengthrestdataset1(nbdatapointdataset1)=pathlengthrestsbj(ii);
            totfixdurarea1dataset1(nbdatapointdataset1)=totfixdurarea1sbj(ii);
            totfixdurarea2dataset1(nbdatapointdataset1)=totfixdurarea2sbj(ii);
            totfixdurrestdataset1(nbdatapointdataset1)=totfixdurrestsbj(ii);
            numfixarea1dataset1(nbdatapointdataset1)=numfixarea1sbj(ii);
            numfixarea2dataset1(nbdatapointdataset1)=numfixarea2sbj(ii);
            numfixrestdataset1(nbdatapointdataset1)=numfixrestsbj(ii);
            nbdatapointdataset1=nbdatapointdataset1+1;
            
        elseif ismember(datatotal(ii),dataset2)
            fixdurarea1dataset2(nbdatapointdataset2)=fixdurarea1sbj(ii);
            fixdurarea2dataset2(nbdatapointdataset2)=fixdurarea2sbj(ii);
            fixdurrestdataset2(nbdatapointdataset2)=fixdurrestsbj(ii);
            pathlengtharea1dataset2(nbdatapointdataset2)=pathlengtharea1sbj(ii);
            pathlengtharea2dataset2(nbdatapointdataset2)=pathlengtharea2sbj(ii);
            pathlengthrestdataset2(nbdatapointdataset2)=pathlengthrestsbj(ii);
            totfixdurarea1dataset2(nbdatapointdataset2)=totfixdurarea1sbj(ii);
            totfixdurarea2dataset2(nbdatapointdataset2)=totfixdurarea2sbj(ii);
            totfixdurrestdataset2(nbdatapointdataset2)=totfixdurrestsbj(ii);
            numfixarea1dataset2(nbdatapointdataset2)=numfixarea1sbj(ii);
            numfixarea2dataset2(nbdatapointdataset2)=numfixarea2sbj(ii);
            numfixrestdataset2(nbdatapointdataset2)=numfixrestsbj(ii);
            nbdatapointdataset2=nbdatapointdataset2+1;
        end
    end
    
    eyeareadataset1=[fixdurarea1dataset1' fixdurarea2dataset1' fixdurrestdataset1' pathlengtharea1dataset1' pathlengtharea2dataset1' pathlengthrestdataset1' totfixdurarea1dataset1' totfixdurarea2dataset1' totfixdurrestdataset1' numfixarea1dataset1' numfixarea2dataset1' numfixrestdataset1'];
    eyeareadataset2=[fixdurarea1dataset2' fixdurarea2dataset2' fixdurrestdataset2' pathlengtharea1dataset2' pathlengtharea2dataset2' pathlengthrestdataset2' totfixdurarea1dataset2' totfixdurarea2dataset2' totfixdurrestdataset2' numfixarea1dataset2' numfixarea2dataset2' numfixrestdataset2'];
    dlmwrite('eyeareadataset1.txt', eyeareadataset1, 'delimiter', '\t', 'precision', 7);
    dlmwrite('eyeareadataset2.txt', eyeareadataset2, 'delimiter', '\t', 'precision', 7);
end