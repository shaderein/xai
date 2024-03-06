function y = changeMouseListCell(x, opt)
    if ~isempty(x)
        selY = x(:,2) - opt.yOffset; selY(selY==0) = 1; selY(selY>opt.ySize) = opt.ySize; 
        selX = x(:,1); selX(selX==0) = 1; selX(selX>opt.xSize) = opt.xSize;
        %     y = [selX selY];
        m = false(opt.ySize, opt.xSize);
        m(selY, selX) = true;
        y = m;
    else
        y = false(opt.ySize, opt.xSize);
    end

end