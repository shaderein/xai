        function strRes = mat2string(x)
            if ~isempty(x) && ~isequal(x,0)
                x_str = string(x);
                x_str_13 = string(arrayfun(@(x) strcat(x,","), x_str(:,1:3), 'UniformOutput', false));
                x_str_1 = string(arrayfun(@(x) strcat("[",x), x_str_13(:,1), 'UniformOutput', false));
                x_str_4 = string(arrayfun(@(x) strcat(x,"]."), x_str(:,4), 'UniformOutput', false));
                x_str_new = [x_str_1 x_str_13(:,2:end) x_str_4];
                x_str_new = x_str_new';
                strRes = horzcat(x_str_new{:});
        
            else
                strRes = [];
            end
    
        end