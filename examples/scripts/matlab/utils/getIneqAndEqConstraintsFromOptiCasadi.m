% --------------------------------------------------------------------------
% Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
% See LICENSE file for the license information
% -------------------------------------------------------------------------- 

%returns the constraints as 
%    inequality_expressions<=0
%    equality_expressions=0
function [inequality_expressions, equality_expressions]=getIneqAndEqConstraintsFromOptiCasadi(opti_tmp)

    g=opti_tmp.g();
    lower=opti_tmp.lbg();
    upper=opti_tmp.ubg();

    inequality_expressions=[];
    equality_expressions=[];    
    
    %constraints are lower<=g<=upper
    
    all_g=[];
    all_upper=[];
    
    for i=1:size(g,1)

        u_i=upper(i);
        l_i=lower(i);
        g_i=g(i);

        if(abs(convertMX2Matlab(u_i-l_i))<1e-7) %Equality constraint
            equality_expressions=[equality_expressions; g_i-u_i];             % l_i<=g_i<=u_i (with l_i==u_i) <-->  g_i-u_i = 0
            continue
        end

        if(isPlusInfCasadi(u_i)==false)
            inequality_expressions=[inequality_expressions; g_i-u_i];        % g_i<=u_i   <-->  g_i-u_i <=0
        end
        if(isMinusInfCasadi(l_i)==false)
            inequality_expressions=[inequality_expressions; l_i-g_i];        % l_i<=g_i   <-->  l_i-g_i <=0
        end
    end

    % The constraints are now 
    %    inequality_expressions<=0
    %    equality_expressions==0
    

end