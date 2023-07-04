% --------------------------------------------------------------------------
% Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
% See LICENSE file for the license information
% -------------------------------------------------------------------------- 

function result=randInInterval(a,b,s)
    result=[a + (b-a).*rand(s(1),s(2))];
end