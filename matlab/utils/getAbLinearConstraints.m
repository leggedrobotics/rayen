%opti_tmp is an optimization problem from Casadi
%Feasible set is Ax<=b
%Note that en equality constraint Aeq=b is equivalent to Aeq<=b Aeq>=b
%The equality constraints are also returned in [A,b] using the transformation explained in the above line
function [A,b]=getAbLinearConstraints(opti_tmp)

    g=opti_tmp.g();
    lower=opti_tmp.lbg();
    upper=opti_tmp.ubg();
    
    %constraints are lower<=g<=upper
    
    all_g=[];
    all_upper=[];
    
    for i=1:size(g,1)
        if(isPlusInfCasadi(upper(i))==false)
    %         upper(i)
            all_upper=[all_upper; upper(i)];
            all_g=[all_g; g(i)];
        end
        if(isMinusInfCasadi(lower(i))==false)
            all_upper=[all_upper; -lower(i)];
            all_g=[all_g; -g(i)];
        end
    end
    % The constraints are now all_g<=all_upper


    % The constraints are now all_g_minus_all_upper<=0
    all_g_minus_all_upper=all_g-all_upper;
    
    variables=opti_tmp.x;

    b=-casadi.substitute(all_g_minus_all_upper, variables, zeros(size(variables))); %Note the - sign
    A=jacobian(all_g_minus_all_upper, variables);
    try
        b=convertMX2Matlab(b);
        A=convertMX2Matlab(A);
    catch
        error('Are you sure you only have linear constraints?');
    end

end