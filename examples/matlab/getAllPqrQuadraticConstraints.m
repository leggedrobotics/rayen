%opti_tmp is an optimization problem from Casadi
%Feasible set is 0.5*x'P_i x + q'x + r <=0
function [all_P, all_q, all_r]=getAllPqrQuadraticConstraints(opti_tmp)
    
     [inequality_expressions, equality_expressions]=getIneqAndEqConstraintsFromOptiCasadi(opti_tmp);


    if(numel(equality_expressions)>0)
        error("Not implemented yet")
    end

    variables=opti_tmp.x;

    all_P={};
    all_q={};
    all_r={};

    %%%%%% INEQUALITY CONSTRAINTS
    for i=1:size(inequality_expressions,1)
        i
        [P,q,r]=getPandqandrOfQuadraticExpressionCasadi(inequality_expressions(i), variables);

        try
            P=convertMX2Matlab(P);
            q=convertMX2Matlab(q);
            r=convertMX2Matlab(r);
        catch %Constraints are not linear
            error('Are you sure you only have quadratic constraints?');
        end

        all_P{end+1}=P;
        all_q{end+1}=q;
        all_r{end+1}=r;


    end
    %%%%%%%%%%%%%%%%%%%%%%%%%

end