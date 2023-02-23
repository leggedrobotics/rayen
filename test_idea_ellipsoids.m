close all; clc;clear;
addpath(genpath('./deep_panther/panther/matlab'))
addpath(genpath('./deep_panther/submodules/minvo'))
addpath(genpath('./utils'))
V1=[0 5 1 0;
    0 0 1 1];

[A,b,Aeq,beq]=vert2lcon(V1',1e-10);

A(abs(A)<1e-10) = 0;

plot2dConvHullAndVertices(V1)
x0=[2;0.6];
E=computeDikinEllipsoid(A,b,x0);
plot_ellipseE_and_center(E, x0)


[B,x0]=convertErepresentation2Brepresentation(E,x0);
plot_ellipseB(B,x0)

%E representation --> {x s.t. (x-x0)'E(x-x0) <= 1}
%B representation --> {x s.t. x=B*p_bar + x0, ||p_bar||<=1} \equiv {x s.t. ||inv(B)(x-x0)||<=1} \equiv {x s.t. (x-x0)'*inv(B)'*inv(B)*(x-x0)<=1}
%More info about the B representation: https://ieeexplore.ieee.org/abstract/document/7839930
function [B,x0]=convertErepresentation2Brepresentation(E,x0)
    
    %See %https://laurentlessard.com/teaching/cs524/slides/11%20-%20quadratic%20forms%20and%20ellipsoids.pdf
    B_inv=sqrtm(E);
    B=inv(B_inv)'

end

%The ellipsoid has the equation (x-x0)'E(x-x0) <= 1
%Polyhedron is defined by Ax<=b
function E=computeDikinEllipsoid(A,b,x0)

     %Dikin ellipsoid is defined by {y | (y-x)'E(y-x)<=1}
     % (see slide 13-7 of http://www.seas.ucla.edu/~vandenbe/ee236a/lectures/cpath.pdf)
    rows_A=size(A,1);
    d_x=ones(rows_A,1)./(b-A*x0); %Slide 13-5
    E=A'*diag(d_x)^2*A;

end

%plots an ellipse using the B representation
function plot_ellipseB(B,x0)
     theta=0:0.1:2*pi;
     r=1.0;
     x_circle=[r*cos(theta);
               r*sin(theta)]; %This the unit circle

     x_ellipse=B*x_circle + repmat(x0,1,numel(theta));
     plot(x_ellipse(1,:),x_ellipse(2,:),'o');
end

%plots an ellipse using the E representation
function plot_ellipseE(E,x0)

 %FIRST WAY
 x= sym('x', [numel(x0) 1]);
 fimplicit((x-x0)'*E*(x-x0)-1)

 %SECOND WAY
 %Taken from https://stackoverflow.com/a/34308544/6057617
 R = chol(E);
 t = linspace(0, 2*pi, 100); % or any high number to make curve smooth
 z = [cos(t); sin(t)];
 ellipse = inv(R) * z;
 ellipse = ellipse + repmat(x0,1,size(ellipse,2));
 plot(ellipse(1,:), ellipse(2,:),'--r')
end 

function plot_ellipseE_and_center(E,x0)

    plot_ellipseE(E,x0);
    plot(x0,'*');

end


% expression=(y-x_test)'*H_test*(y-x_test)-1;
% expression=subs(expression,x,x_test);

% plot_ellipse(subs(H,x,x_test),x_test)
% syms x1 x2
% x=[x1;x2];
% 
% 
% %First way
% figure;
% plot_ellipse(H_test)
% 
% %Second way
% fimplicit(x'*H_test*x-1,'--r')

% close all; figure;
% H_test=double(subs(H,x,[2;3]))
% plot_ellipse(H_test,[1;2])


% [V D]=eig(H_test);
% tmp=diag(D);
% semiaxes=1./sqrt(tmp);
% 
% % function plotEllipseFromHessian()
% 
% phi=0:0.01:2*pi;
% x=semiaxes(1)*cos(phi);
% y=semiaxes(2)*sin(phi);
% 
% [x ; y];
% 
%     
% % end