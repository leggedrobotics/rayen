% --------------------------------------------------------------------------
% Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
% See LICENSE file for the license information
% -------------------------------------------------------------------------- 

%opti_tmp is an optimization problem from Casadi
%Feasible set is A1x<=b1 \cap A2x=b2 
function [A1,b1, A2, b2]=getAbLinearConstraints(opti_tmp)
    
   [inequality_expressions, equality_expressions]=getIneqAndEqConstraintsFromOptiCasadi(opti_tmp);

    variables=opti_tmp.x;

    %%%%%%%%%% INEQUALITY CONSTRAINTS
    b1=-casadi.substitute(inequality_expressions, variables, zeros(size(variables))); %Note the - sign
    A1=jacobian(inequality_expressions, variables);
    try
        A1=convertMX2Matlab(A1);
        b1=convertMX2Matlab(b1);
    catch %Constraints are not linear
        error('Are you sure you only have linear constraints?');
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %%%%%%%%%% EQUALITY CONSTRAINTS
    b2=-casadi.substitute(equality_expressions, variables, zeros(size(variables))); %Note the - sign
    A2=jacobian(equality_expressions, variables);
    try
        A2=convertMX2Matlab(A2);
        b2=convertMX2Matlab(b2);
    catch %Constraints are not linear
        error('Are you sure you only have linear constraints?');
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end