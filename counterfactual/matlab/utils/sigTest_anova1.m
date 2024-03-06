function p4 = sigTest_anova1(x1,x2)
x1 = x1(:);
x2 = x2(:);
x1_label = ones(size(x1));
x2_label = 2*ones(size(x2));
x = cat(1, x1, x2);
x_label = cat(1, x1_label, x2_label);

% [num,txt,raw] = xlsread('Resting State.xlsx');
%  
%% indenpendent two sample ttest
p4 = anova1(x, x_label);


end

