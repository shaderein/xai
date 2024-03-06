function T = replaceNaNWithZero(T)
    % 检查输入是否为表格
    if ~istable(T)
        error('输入必须为表格类型变量');
    end

    % 获取表格的列名
    varNames = T.Properties.VariableNames;

    % 遍历所有列
    for i = 1:numel(varNames)
        % 检查列的数据类型
        if isnumeric(T.(varNames{i}))
            % 如果是数值类型，用0替换NAN
            T.(varNames{i})(isnan(T.(varNames{i}))) = 0;
        end
    end
end