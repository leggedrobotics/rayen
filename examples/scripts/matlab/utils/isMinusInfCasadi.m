% --------------------------------------------------------------------------
% Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
% See LICENSE file for the license information
% -------------------------------------------------------------------------- 

function result=isMinusInfCasadi(expression)
    if(containsSymCasadi(expression)) %if it has symbolic variables
        result=false;
        return;
    end
    result=(convertMX2Matlab(expression)==-inf);
end