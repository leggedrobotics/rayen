close all; clc;clear;
set(0,'DefaultFigureWindowStyle','docked')%normal or docked
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


% F=inv(diag((b-A*x0)));
% x= sym('x', [numel(x0) 1]);
% FA=F*A;
% fimplicit((x-x0)'*(FA')*FA*(x-x0)-1,'*')



all_points_in_polytope=[];

Ax_k=A*x0;

for i=1:100
[E,F,F_inv]=computeDikinEllipsoidGivenAx0(A,b,Ax_k)
sample_ball=uniformSampleInUnitBall(4,1);
Ax_kp1=F_inv*sample_ball + Ax_k

assert(all(Ax_kp1<=b)) %Assert that we are still inside the polytope

%Just for visualization
point_in_polytope=linsolve(A,Ax_kp1);

all_points_in_polytope=[all_points_in_polytope point_in_polytope];

end

plot(all_points_in_polytope(1,:),all_points_in_polytope(2,:),'o')

%     for j=1:10
%         [E,F,F_inv]=computeDikinEllipsoidGivenAx0(A,b,Ax_k)
%         sample_ball=uniformSampleInUnitBall(4,1);
%         Ax_kp1=F_inv*sample_ball + Ax_k;
%         
% %         Ax_k=Ax_kp1;
%         
%         assert(all(Ax_kp1<=b)) %Asser that we are still inside the polytope
%         
%         %Just for visualization
%         point_in_polytope=linsolve(A,Ax_kp1);
% 
% %         assert(all(A*point_in_polytope<=b))
%         
%         all_points_in_polytope=[all_points_in_polytope point_in_polytope];
%     end
% 
% plot(all_points_in_polytope(1,:),all_points_in_polytope(2,:),'o')

%%
figure; hold on;axis equal;
samples=uniformSampleInUnitBall(2,4000);
plot(samples(1,:),samples(2,:),'o')

%%
% [E, F, F_inv]=computeDikinEllipsoidGivenAx0(A,b,Ax_k)


function result=uniformSampleInUnitBall(dim,num_points)
%Method 20 of http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

u = normrnd(0,1,dim,num_points);  % each column is an array of dim normally distributed random variables
u_normalized=u./vecnorm(u);
r = rand(1,num_points).^(1.0/dim); %each column is the radius of each of the points
result= r.*u_normalized;

end


%E representation --> {x s.t. (x-x0)'E(x-x0) <= 1}. Here, E is a psd matrix
%B representation --> {x s.t. x=B*p_bar + x0, ||p_bar||<=1} \equiv {x s.t. ||inv(B)(x-x0)||<=1} \equiv {x s.t. (x-x0)'*inv(B)'*inv(B)*(x-x0)<=1}. 
%B is \in R^nxn
%More info about the B representation: https://ieeexplore.ieee.org/abstract/document/7839930
function [B,x0]=convertErepresentation2Brepresentation(E,x0)
    
    %See https://laurentlessard.com/teaching/cs524/slides/11%20-%20quadratic%20forms%20and%20ellipsoids.pdf
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

function [E, F, F_inv]=computeDikinEllipsoidGivenAx0(A,b,Ax0)
    rows_A=size(A,1);
    bminusAx0=b-Ax0;
    
    d_x=ones(rows_A,1)./(bminusAx0); %Slide 13-5
    F=diag(d_x);

    F_inv=diag(bminusAx0);

    E=A'*F^2*A;
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