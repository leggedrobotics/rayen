close all; clc;clear;
set(0,'DefaultFigureWindowStyle','docked')%normal or docked
addpath(genpath('./../deep_panther/panther/matlab'))
addpath(genpath('./../deep_panther/submodules/minvo'))
addpath(genpath('./utils'))

center=[0.5,0.5,0.5];
side=[1.0 1.0 1.0];
box =getAb_Box3D(center,side);
A=box.A;
b=box.b;
[V,nr,nre]=lcon2vert(A,b,[],[]);
V=V'; %my convention
Aeq=[1 1 1;
     -1 1 1];
beq=[1;0.1];


% Aeq=[1 0 0];
% beq=0.0;

plot3dConvHullAndVertices(V, 0.02)

syms x y z
for i=1:size(Aeq,1)
    fimplicit3(Aeq(i,:)*[x;y;z]-beq(i,1),[-1 1 -1 1 -1 1],'EdgeColor','none','FaceAlpha',.5)
end


z = sdpvar(size(A,2),1);


A=[A;Aeq;-Aeq];
b=[b;beq;-beq];

%%Remove constraints that are redundant
%Eq 1.5 of https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/167108/1/thesisFinal_MaySzedlak.pdf
%See also https://mathoverflow.net/a/69667
redundant_set=[];
all_objectives=[];
for i=size(A,1):-1:1
    objective=-A(i,:)*z; %Note that we are maximizing A(i,:)*z
    tmp=setdiff(1:size(A,1),i); %all the rows except the i-th one
    constraints=[A(tmp,:)*z<=b(tmp)    A(i,:)*z<=(b(i)+1)]; %The +1 prevents the problem to become unbounded because of the change in the constraints
    optimize(constraints,objective,sdpsettings('solver','mosek','verbose',false));
    obj=value(objective)
    all_objectives=[all_objectives obj];
    
    if((-obj)<=b(i)) %Note that I need -obj because we are maximizing
        disp("deleting")
        i
        A(i,:)=[]; %Note that I need to do it here (instead of batched at the end) because if there are two constraints that are equal --> don't wanna delete both
        b(i,:)=[];
    end
end

%%Now let's find the equality set
% Section 5.2 of https://www.researchgate.net/publication/268373838_Polyhedral_Tools_for_Control
% Eq. 1.5 of https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/167108/1/thesisFinal_MaySzedlak.pdf
% See also Definition 2.16 of https://sites.math.washington.edu/~thomas/teaching/m583_s2008_web/main.pdf
z = sdpvar(size(A,2),1);
equality_set=[];
all_objectives=[];

for i=1:size(A,1)

    objective=(A(i,:)*z-b(i));
    constraints=[A*z<=b];
    s=optimize(constraints,objective,sdpsettings('solver','mosek'));
    obj=value(objective);
    all_objectives=[all_objectives obj]
    
    %https://yalmip.github.io/command/optimize/
    if s.problem == 0
    %  disp('Solver thinks it is feasible')
    elseif s.problem == 1
     error('Solver thinks it is infeasible')
    else
     error('Something else happened')
    end
    
    tol=1e-6;
    assert(obj<=tol,"obj is %f",obj)
    if(obj>-tol) %objective is zero. %Note that the objective is always negative
        equality_set=[equality_set i];
    end
end


%remove the ones that are in the equality set
A_eq_set=A(equality_set,:);
b_eq_set=b(equality_set,:);
A(equality_set,:)=[];
b(equality_set,:)=[];

NA_eq_set=null(A_eq_set);
x0=pinv(A_eq_set)*b_eq_set;
bbb=b-A*x0;
AAA=A*NA_eq_set;


