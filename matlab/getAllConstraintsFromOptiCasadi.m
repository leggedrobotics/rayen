%returns the constraints as expression<=0
function all_g_minus_all_upper=getAllConstraintsFromOptiCasadi(opti_tmp)

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


    
    all_g_minus_all_upper=all_g-all_upper;
    % The constraints are now all_g_minus_all_upper<=0

end