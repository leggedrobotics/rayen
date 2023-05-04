%opti_tmp is an optimization problem from Casadi
%Feasible set is Ax<=b
%Note that en equality constraint Aeq=b is equivalent to Aeq<=b Aeq>=b
%The equality constraints are also returned in [A,b] using the transformation explained in the above line
function [A,b]=getAbLinearConstraints(opti_tmp)
    
    all_g_minus_all_upper=getAllConstraintsFromOptiCasadi(opti_tmp);

    variables=opti_tmp.x;

    b=-casadi.substitute(all_g_minus_all_upper, variables, zeros(size(variables))); %Note the - sign
    A=jacobian(all_g_minus_all_upper, variables);
    try
        b=convertMX2Matlab(b);
        A=convertMX2Matlab(A);
    catch %Constraints are not linear
        error('Are you sure you only have linear constraints?');
    end

end