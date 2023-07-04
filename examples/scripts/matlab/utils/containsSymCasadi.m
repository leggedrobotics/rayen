% --------------------------------------------------------------------------
% Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
% See LICENSE file for the license information
% -------------------------------------------------------------------------- 

function result=containsSymCasadi(expression)
    result=(numel(symvar(expression))>0);
end