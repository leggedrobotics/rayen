close all; clc;clear;
addpath(genpath('./deep_panther/panther/matlab'))
addpath(genpath('./deep_panther/submodules/minvo'))

doSetup();
import casadi.*

const_p={};    const_y={};
opti = casadi.Opti();

t0_n=0.0; 
tf_n=1.0;

basis="MINVO";%MINVO B_SPLINE
deg_pos=3;
num_seg =2; %right now it needs to be equal to the number of obstacles
dim_pos=2;

sp=MyClampedUniformSpline(t0_n,tf_n,deg_pos, dim_pos, num_seg, opti); %spline position.

% center1=[0,0,0];
% center2=[0.5,0.0,0.0];
% center3=[1.5,-0.8,0.0];
% center4=[2.8,0.5,0.0];
% C=[center1',  center2',  center3',  center4'];
% 
% side1=[2.0, 2.0, 2.0];
% side2=[2.0, 2.0, 2.0]; 
% side3=[1.1, 1.8, 1.8]; 
% side4=[1.8, 1.8, 1.8];


% V1_raw=rand(3,4);
% 
% [k1,av1] =convhull(V1_raw(1,:),V1_raw(2,:),V1_raw(3,:));
% k=unique(k1(:));
% V1=V1_raw(:,k);
% V2=V1;
% 
% [A1,b1,Aeq1,beq1]=vert2lcon(V1',1e-10);
% [A2,b2,Aeq2,beq2]=vert2lcon(V2',1e-10);
% 
% polyhedra{1}.A=A1;
% polyhedra{1}.b=b1;
% 
% polyhedra{2}.A=A2;
% polyhedra{2}.b=b2;

% polyhedra{1}=getAb_Box(center1, side1);
% polyhedra{2}=getAb_Box(center2, side2);
%polyhedra{3}=getAb_Box(center3, side3);
%polyhedra{4}=getAb_Box(center4, side4);

% polyhedra{1}=getAb_Box2D([0.5,0.5],[1.0,1.0]);
% polyhedra{2}=getAb_Box2D([1.0,0.5],[1.0,1.0]);

%First polytope
V1=[0 1 1 0;
    0 0 1 1];

%Second polytope
V2=V1+[0.6;0.8];

[A1,b1,Aeq1,beq1]=vert2lcon(V1',1e-10);
[A2,b2,Aeq2,beq2]=vert2lcon(V2',1e-10);

polyhedra{1}.A=A1; polyhedra{1}.b=b1;
polyhedra{2}.A=A2; polyhedra{2}.b=b2;

const_p_obs_avoid={};

for j=1:(sp.num_seg)
    %Get the control points of the interval
    Q=sp.getCPs_XX_Pos_ofInterval(basis, j);
    
    for kk=1:size(Q,2) %for all the control points of the interval
        const_p_obs_avoid{end+1}= polyhedra{j}.A*Q{kk} - polyhedra{j}.b <= 0;
    end
end
% 
opti.subject_to([const_p_obs_avoid]);
[A,b]=getAbLinearConstraints(opti);
Aeq=[]; beq=[]; TOL=[]; %default tolerance, it is 1e-10

[V,nr,nre]=lcon2vert(A,b,Aeq,beq,TOL);
V=V';

figure; hold on;

%%OPTION 1: Check that there exist barycentric coordinates for a fixed spline (which is known to 
%%satisfy the constraints) 
bs_cps=[0.1 0.4 0.8 0.9 1.4;
        0.1 0.6 1.35 0.5 1.5];
num_bary_coor=size(V,2);
f=zeros(num_bary_coor,1);
% A=ones(1,num_bary_coor); b=1; %Ax<=1
A=[]; b=[]; %Ax<=1
Aeq=[V; ones(1,num_bary_coor)]; beq=[bs_cps(:) ; 1]; %Vx=variables and sum(x)=1
lb=zeros(num_bary_coor,1);
ub=ones(num_bary_coor,1);
bar_coord = linprog(f,A,b,Aeq,beq, lb, ub);
assert(all(bar_coord>=0));
assert(abs(sum(bar_coord)-1)<1e-7);
bs_cps=V*bar_coord;

%NOTE: THE "PROBLEM" with options 2 and 3 is that we are sampling from a very high-dimensional 
% simplex. This means that, even if we sample uniformly, most of the samples will fall close to the
% center of the simplex, because there is a lot of volume there.

%OPTION 2:
% bar_coord=sampleBarCoord(1,size(V,2));
% bs_cps=V*bar_coord;

%OPTION 3: (See paper "Geometrically Constrained Traj Opt for Multicopters)
% v0=V(:,1);
% Vhat=V(:,2:end)-repmat(v0,1,size(V,2)-1);
% bar_coord=samplePointsSimplex(1,size(Vhat,2));
% bs_cps=v0 + Vhat*bar_coord;

bs_cps=reshape(bs_cps,sp.dim,[]);

bs_cps=num2cell(bs_cps,1);
sp.setCPoints(bs_cps);

sp.plotPos2D()

    for j=1:(sp.num_seg)
        %Get the control points of the interval
        Q=cell2mat(sp.getCPs_XX_Pos_ofInterval(basis, j));
        scatter(Q(1,:),Q(2,:))        
    end

% end


plot2dConvHullAndVertices(V1);
plot2dConvHullAndVertices(V2);

%%

% close all
% tmp=samplePointsSimplex(10000,3); %Sample points from a simplex
% scatter3(tmp(1,:),tmp(2,:),tmp(3,:),'filled'); axis equal;

%Example in 2D: x>=0, y>=0, x+y=1
function result=sampleBarCoordVersion2(num_points, dim)
    %https://stackoverflow.com/a/67202070
    samples=[];

    for i=1:num_points
        sample=exprnd(1e9,dim,1);
        samples=[samples sample];
    end
    result=samples./sum(samples); %divide by the sum of each column
end

%Example in 2D: x>=0, y>=0, x+y=1
function result=sampleBarCoord(num_points, dim)
    %https://cs.stackexchange.com/a/3229
    samples=[];

    for i=1:num_points
        tmp=[0; rand(dim-1,1); 1];
        sample=diff(sort(tmp));
        samples=[samples sample];
    end
    result=samples;
end

%Example in 2D: x>=0, y>=0, x+y<=1
function result=samplePointsSimplex(num_points, dim)
%     samples=[];
% 
%     for i=1:num_points
%         sample=samplePoints(zeros(dim,1),ones(dim,1),num_points,1,0)';
%         samples=[samples sample];
%     end
%     result=samples;
    result=samplePoints(zeros(dim,1),ones(dim,1),num_points,1,0);
end


function plot2dConvHullAndVertices(V1)
[k,av] = convhull(V1');
plot(V1(1,:),V1(2,:),'*')
hold on
plot(V1(1,k),V1(2,k))
end

function result=getAb_Box2D(center,side)

    A1=[1 0;
        0 1;
        -1 0;
        0 -1];
    
    of_x=center(1);
    of_y=center(2);


    b1=[side(1)/2.0+of_x;
        side(2)/2.0+of_y;
        side(1)/2.0-of_x;
        side(2)/2.0-of_y];
    
    result.A=A1;
    result.b=b1;

end


function result=getAb_Box3D(center,side)

    A1=[1 0 0;
        0 1 0;
        0 0 1;
        -1 0 0;
        0 -1 0;
        0 0 -1];
    
    of_x=center(1);
    of_y=center(2);
    of_z=center(3);
    
    b1=[side(1)/2.0+of_x;
        side(2)/2.0+of_y;
        side(3)/2.0+of_z;
        side(1)/2.0-of_x;
        side(2)/2.0-of_y;
        side(3)/2.0-of_z];
    
    result.A=A1;
    result.b=b1;

end

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

%%
% hold on;
% P=getA_MV(2,[-1,1]);
% sym t
% Pt=P*getT(2,t);
% fplot3(Pt(1),Pt(2),Pt(3),[-1,1],'r'); axis equal;

% hold on;
% trisurf(k1,V1_raw(1,:),V1_raw(2,:),V1_raw(3,:),'FaceColor','cyan','FaceAlpha',0.5)
% scatter3(V1(1,:),V1(2,:),V1(3,:),'filled')
% axis equal

% [V1,nr,nre]=lcon2vert(polyhedra{1}.A,polyhedra{1}.b,Aeq,beq,TOL);
% [V2,nr,nre]=lcon2vert(polyhedra{2}.A,polyhedra{2}.b,Aeq,beq,TOL);

% for i=1:2000

% bar_coord=sampleBarCoord(1,size(V,2));
% bar_coord=(1:numel(bar_coord))'; bar_coord=bar_coord/sum(bar_coord);

