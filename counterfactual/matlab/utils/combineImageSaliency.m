function combinedImage = combineImageSaliency(rawImage, att_smo, saliencyMapAlpha, cmap)
%     saliencyMapAlpha = 0.5;
    curSaliencyMapData = uint8(rescale(att_smo).*255);
%     c = size(cmap, 1);
    curSaliencyMapData = ind2rgb(double(curSaliencyMapData)+1, cmap);
    combinedImage = uint8((curSaliencyMapData*saliencyMapAlpha + double(rawImage)./255*(1-saliencyMapAlpha))*255);


end

