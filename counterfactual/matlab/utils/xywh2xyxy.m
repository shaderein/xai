function p_new = xywh2xyxy(p)
    if numel(p) < 4
        p = zeros(1,4);
    end
    p_new = zeros(size(p),"like",p);
    p_new(:,1) = p(:,1)-p(:,3)./2;
    p_new(:,2) = p(:,2)-p(:,4)./2;
    p_new(:,3) = p(:,1)+p(:,3)./2;
    p_new(:,4) = p(:,2)+p(:,4)./2;

end