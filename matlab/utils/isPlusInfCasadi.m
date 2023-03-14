function result=isPlusInfCasadi(expression)
    if(containsSymCasadi(expression)) %if it has symbolic variables
        result=false;
        return;
    end
    result=(convertMX2Matlab(expression)==inf);
end