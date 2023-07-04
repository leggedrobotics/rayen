% --------------------------------------------------------------------------
% Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
% See LICENSE file for the license information
% -------------------------------------------------------------------------- 

%Quadratic expression is (1/2)*x'*P*x + q'*x + r
function [P, q, r] = getPandqandrOfQuadraticExpressionCasadi(my_expression, x)

    import casadi.*

    assert(size(x,2)==1)

    my_gradient=gradient(my_expression,x);
    my_hessian=hessian(my_expression, x);
    
    P=convertMX2Matlab(my_hessian);
    q=convertMX2Matlab(casadi.substitute(my_gradient,x, zeros(size(x))));
    r=convertMX2Matlab(casadi.substitute(my_expression,x, zeros(size(x))));
    

    %Quick check
    x_random_values=10*rand(size(x));
    expression_tmp=0.5*x_random_values'*P*x_random_values + q'*x_random_values + r;
    should_be_zero=convertMX2Matlab(expression_tmp-casadi.substitute(my_expression,x,x_random_values));
    assert(abs(should_be_zero)<1e-5)
end