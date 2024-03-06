function p4 = sigTest_indttest(x1,x2)
x1 = x1(:);
x2 = x2(:);
x1_label = ones(size(x1));
x2_label = 2*ones(size(x2));
x = cat(1, x1, x2);
x_label = cat(1, x1_label, x2_label);

% [num,txt,raw] = xlsread('Resting State.xlsx');
%  
%% indenpendent two sample ttest
% idx=num(:,5);
% x=num(:,1);
% x_M=x(idx==1);
% x_F=x(idx==0);
% 方差齐性检验，即检验两组样本的总体方差是否相同
[p3,stats3] = vartestn(x,x_label,...
    'TestType','LeveneAbsolute','Display','off');
disp('Independent t-test with Eyes open:');
disp(['Levene’s test: p = ',num2str(p3,'%0.2f')]);%方差检验方法：Levene检验
if p3<0.05
    disp('Equal variances not assumed') %方差不相同
    [h4,p4,ci4,stats4]=ttest2(x1,x2,...
        'Vartype','unequal');                                 
else
    disp('Equal variances assumed'); %方差相同
    [h4,p4,ci4,stats4]=ttest2(x1,x2);
end
disp(['t = ',num2str(stats4.tstat,'%0.2f')]);
disp(['df = ',num2str(stats4.df,'%0.2f')]);
disp(['p = ',num2str(p4,'%0.2f')]);

end

