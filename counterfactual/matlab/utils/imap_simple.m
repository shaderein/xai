function output_map = imap_simple(cfg)

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
if length(datatotal) > 1
    warning('Dataset > 1');

end

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
    output_map = smoothpic;

end


end




