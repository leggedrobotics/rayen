function result=containsSymCasadi(expression)
    result=(numel(symvar(expression))>0);
end