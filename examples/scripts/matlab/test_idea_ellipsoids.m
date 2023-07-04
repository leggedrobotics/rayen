% --------------------------------------------------------------------------
% Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
% See LICENSE file for the license information
% -------------------------------------------------------------------------- 

close all; clc;clear;
set(0,'DefaultFigureWindowStyle','docked')%normal or docked
addpath(genpath('./deep_panther/panther/matlab'))
addpath(genpath('./deep_panther/submodules/minvo'))
addpath(genpath('./utils'))
V1=[0 5 1 0;
    0 0 1 1];

[A,b,Aeq,beq]=vert2lcon(V1',1e-10);

A(abs(A)<1e-10) = 0;

x0=[1.8;0.4];
plot2dConvHullAndVertices(V1)

% E=computeDikinEllipsoid(A,b,x0);
% plot_ellipseE_and_center(E, x0)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Largest ellipsoid in polytope
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=2;
B = sdpvar(n,n); %Symmetric
x0 = sdpvar(n,1);
constraints=[];
%Eq. 8.15 of https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
%Also http://web.cvxr.com/cvx/examples/cvxbook/Ch08_geometric_probs/html/max_vol_ellip_in_polyhedra.html
for i=1:size(A,1)
    a_i=A(i,:)';
    b_i=b(i);
    constraints=[constraints norm(B*a_i)+a_i'*x0 <=b_i];
end
optimize(constraints,-logdet(B),sdpsettings('solver','mosek')) %mosek
B=value(B);
x0=value(x0);
plot_ellipseB(B,x0);

% scaleEllipsoidB(B,A,b,[0.5;0.2])

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% x0=[3;0.2];
% B=scaleEllipsoidB(B,A,b,x0);
% plot_ellipseB(B,x0);

x0_original=x0;

all_x0=[];
for i=1:7000
    x0=x0_original;
    for j=1:10
        %Note: if you want to sample uniformly from the ellipsoid, then you need a cholescky decomposition: https://github.com/stla/uniformly/blob/master/R/ellipsoid.R
        B=scaleEllipsoidB(B,A,b,x0);
        u=uniformSampleInUnitSphere(2,1); %random unit vector 
        
%         %Using the E representation
%         lambda=1/sqrt(u'*E*u);%This is the distance from the center of the ellipsoid to the border of the ellipsoid in the direction of u
%         length=lambda*rand();
%         x0_new=x0+length*u;
% 
%         %Using the B representation
%         %Option 1: x0_new will be in the direction of u
%         B_inv=inv(B);
%         lambda=1/sqrt(u'*B_inv'*B_inv*u);%This is the distance from the center of the ellipsoid to the border of the ellipsoid in the direction of u
%         length=lambda*rand();
%         x0_new=x0+length*u;

        %Option 2:x0_new will NOT be in the direction of u
        u_ball=uniformSampleInUnitBall(2,1); %random vector in ball
        x0_new=B*u_ball + x0; 

        x0=x0_new;

        all_x0=[all_x0 x0];
    end
end

plot(all_x0(1,:),all_x0(2,:),'o')

%%

all_x0=[x0];

    E=computeDikinEllipsoid(A,b,x0);
    plot_ellipseE_and_center(E, x0)

    E=scaleEllipsoidE(E,A,b,x0)
    plot_ellipseE_and_center(E, x0)

%     new_x0=x0;
for i=1:1000
     new_x0=x0;
    for j=1:1
        %Note: if you want to sample uniformly from the ellipsoid, then you need a cholescky decomposition: https://github.com/stla/uniformly/blob/master/R/ellipsoid.R
        E=computeDikinEllipsoid(A,b,x0);
        scaleEllipsoidE(E,A,b,x0)
        u=uniformSampleInUnitSphere(2,1); %random unit vector 
%         u= -1 + 2.*rand(2,1); %random vector in [-1,1]^2
%         u=normalize(u);
        lambda=1/sqrt(u'*E*u);%This is the distance from the center of the ellipsoid to the border of the ellipsoid in the direction of u
        length=lambda*rand()
        new_x0=x0+length*u;
    
        all_x0=[all_x0 new_x0];
    %     plot(point(1),point(2),'o')
    %     my_arrow=[x0 x0+u];
    %     plot(my_arrow(1,:),my_arrow(2,:))
    end
end

plot(all_x0(1,:),all_x0(2,:),'o')

%%
%c=sym('c',[4,1])
% eig(A'*diag(c)*A)
% [E,D]=computeDikinEllipsoid(A,b,x0);
% 
% S = svd(A)
% [U,S,V] = svd(A);
% [Q,Lambda] = eig(A'*D*A); %A'DA=QLambdaQ'
% 
% V*S'*U'*D*U*S*V'

%%
%%
[B,x0]=convertErepresentation2Brepresentation(E,x0);
plot_ellipseB(B,x0)


F=inv(diag((b-A*x0)));
x= sym('x', [numel(x0) 1]);
FA=F*A;
fimplicit((x-x0)'*(FA')*FA*(x-x0)-1,'*')



all_points_in_polytope=[];

Ax_k=A*x0;

for i=1:100
[E,F,F_inv]=computeDikinEllipsoidGivenAx0(A,b,Ax_k)
sample_ball=uniformSampleInUnitBall(4,1);
Ax_kp1=F_inv*sample_ball + Ax_k

assert(all(Ax_kp1<=b)) %Assert that we are still inside the polytope

%Just for visualization
point_in_polytope=linsolve(A,Ax_kp1)
inside_polytope=A*point_in_polytope<=b
assert(all(inside_polytope))

all_points_in_polytope=[all_points_in_polytope point_in_polytope];

end

plot(all_points_in_polytope(1,:),all_points_in_polytope(2,:),'o')


%%
%Want to solve A*x=c
x=linsolve(A,c)%This system does not have a solution --> linsolve returns the least squares solution
A*x-c
lsqr(A,c)

% linsolve([1;1],[2;5])

%%

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
figure; hold on;axis equal;
samples=uniformSampleInUnitBall(3,4000);
plot3(samples(1,:),samples(2,:),samples(3,:),'o'); axis equal;
% sphere

%%
% [E, F, F_inv]=computeDikinEllipsoidGivenAx0(A,b,Ax_k)



function B=scaleEllipsoidB(B,A,b,x0)
    %See https://math.stackexchange.com/questions/340233/transpose-of-inverse-vs-inverse-of-transpose
    %Note that here we have an ellipsoid not centered in the origin. 
    minimum=Inf;
    for i=1:numel(b)
        a_i=A(i,:)';
        tmp=(b(i)-a_i'*x0)^2/(a_i'*B*B'*a_i); %Note that inv_E:=B*B'
        minimum=min(minimum,tmp);
    end

    B=B*sqrt(minimum);
end


function E=scaleEllipsoidE(E,A,b,x0)
    %See https://math.stackexchange.com/questions/340233/transpose-of-inverse-vs-inverse-of-transpose
    %Note that here we have an ellipsoid not centered in the origin. 
    minimum=Inf;
    inv_E=inv(E);
    for i=1:numel(b)
        a_i=A(i,:)';
        tmp=(b(i)-a_i'*x0)^2/(a_i'*inv_E*a_i);
        minimum=min(minimum,tmp);
    end

    E=E/minimum;

end

function result=normalize(u)
    result=u/norm(u);
end






%E representation --> {x s.t. (x-x0)'E(x-x0) <= 1}. Here, E is a psd matrix
%B representation --> {x s.t. x=B*p_bar + x0, ||p_bar||<=1} \equiv {x s.t. ||inv(B)(x-x0)||<=1} \equiv {x s.t. (x-x0)'*inv(B)'*inv(B)*(x-x0)<=1}. 
%B is \in R^nxn (although Boyd's book says we can assume B is psd (and therefore also symmetric) without loss of generality, see section 8.4.1
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
     theta=0:0.01:2*pi;
     r=1.0;
     x_circle=[r*cos(theta);
               r*sin(theta)]; %This the unit circle

     x_ellipse=B*x_circle + repmat(x0,1,numel(theta));
     plot(x_ellipse(1,:),x_ellipse(2,:),'-');
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