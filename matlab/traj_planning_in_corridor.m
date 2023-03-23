%This file solves the traj planning problem of generating a smooth
%trajectory passing through several 3d polytopes. 

%Note that in some cases (depending on the geometry of the convex corridor), the Bezier basis can be better than the MINVO
%basis (due to the fact that the Bezier basis includes the initial and
%final points --> better job in the overlapping areas between several
%polyhedra

close all; clc;clear;
doSetup();

P=[0 1 2 3 4 3 2;
   0 1 1 2 4 4 4;
   0 1 1 1 4 1 0];
allA={};
allb={};
allV={};

steps=2;
samples_per_step=5;
radius=1.3;

for i=1:(size(P,2)-1)
%     [A, b]=getABgivenP1P2(P(:,i),P(:,i+1));
    [A, b, V]=getAbVerticesPolyhedronAroundP1P2(P(:,i),P(:,i+1), steps, samples_per_step, radius);
    allA{end+1}=A;
    allb{end+1}=b;
    allV{end+1}=V;
end

p0=0.8*P(:,1) + 0.2*P(:,2);
pf=0.2*P(:,end-1) + 0.8*P(:,end);

figure;
hold on;
alpha=0.2;

for i=1:size(allA,2)
    plotregion(-allA{i},- allb{i}, [], [],'r',alpha)
end

% plotPolyhedron(P,'r')
camlight
lighting phong

delta=1.0;
xlim([min(P(1,:))-delta,max(P(1,:))+delta]);
ylim([min(P(2,:))-delta,max(P(2,:))+delta]);
zlim([min(P(3,:))-delta,max(P(3,:))+delta]);

plotSphere(p0,0.08,'g')
plotSphere(pf,0.08,'r')

%%


t0=0.0;
tf=10.0;

num_of_regions=size(allA,2);
num_of_seg_per_region=3; %Note: IF YOU FIND THE ERROR "Matrix product with incompatible dimensions. Lhs is 3x1 and rhs is 3x3." when changing, this, the cause if the hand-coded "n_int_knots=15; " in computeMatrixForClampedUniformBSpline.m. Increase it.

basis="MINVO"; %MINVO OR B_SPLINE or BEZIER. This is the basis used for collision checking (in position, velocity, accel and jerk space), both in Matlab and in C++
linear_solver_name='ma27'; %mumps [default, comes when installing casadi], ma27, ma57, ma77, ma86, ma97 
my_solver='ipopt' %'ipopt' %'gurobi'
print_level=0; %From 0 (no verbose) to 12 (very verbose), default is 5

if (strcmp(my_solver,'gurobi'))
opti = casadi.Opti('conic');%'conic' I think you need to use 'conic' for gurobi
else
 opti = casadi.Opti();
end
deg_pos=3;
dim_pos=3;
num_seg =num_of_seg_per_region*num_of_regions;
sp=MyClampedUniformSpline(t0,tf,deg_pos, dim_pos, num_seg, opti);

v0=zeros(3,1); a0=zeros(3,1);
vf=zeros(3,1); af=zeros(3,1);

v_max=4*ones(3,1);   a_max=6*ones(3,1);   j_max=50*ones(3,1);

%Initial conditions
opti.subject_to( sp.getPosT(t0)== p0);
opti.subject_to( sp.getVelT(t0)== v0);
opti.subject_to( sp.getAccelT(t0)== a0);

%Final conditions
% opti.subject_to( sp.getPosT(tf)== pf);
opti.subject_to( sp.getVelT(tf)== vf);
opti.subject_to( sp.getAccelT(tf)== af);


% all_binary={};

const_p={}
const_p=[const_p sp.getMaxVelConstraints(basis, v_max)];      %Max vel constraints (position)
const_p=[const_p sp.getMaxAccelConstraints(basis, a_max)];    %Max accel constraints (position)
const_p=[const_p sp.getMaxJerkConstraints(basis, j_max)];     %Max jerk constraints (position)
opti.subject_to([const_p]);

%Corridor constraints
for j=1:(sp.num_seg)
    
     Q=sp.getCPs_XX_Pos_ofInterval(basis, j);%Get the control points of the interval
    
    %%%%%To force the interval allocation:
    ip=ceil(j/num_of_seg_per_region); %ip is the index of the polyhedron
    
    for kk=1:size(Q,2)
            opti.subject_to( allA{ip}*Q{kk}<=allb{ip}); %Each segment must be in each corridor
    end
    
    %%%%%To use binary variables for the interval allocation
%     tmp=opti.variable(num_of_regions,1);
%     
%     M=1000;
%     for ip=1:num_of_regions
%         for kk=1:size(Q,2)
%                 opti.subject_to( allA{ip}*Q{kk}<=allb{ip}+M*(1-tmp(ip))); %Each segment must be in each corridor
%         end
%     end
%     
%     opti.subject_to(sum(tmp)>=1);
%     
%     all_binary{end+1}=tmp;
    
end



%%%%%%%%%%%%%%%%%% COST
vel_cost=sp.getVelCost();
accel_cost=sp.getAccelCost();
jerk_cost=sp.getControlCost();
final_pos_cost=(sp.getPosT(tf)- pf)'*(sp.getPosT(tf)- pf);

weights=opti.parameter(3,1);

cost=simplify(weights(1)*vel_cost + weights(2)*accel_cost +  weights(3)*jerk_cost + final_pos_cost);
opti.minimize(cost)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts = struct;
% num_of_iv=numel(all_binary)*size(all_binary{1},1); %Num of integer variables
% opts.discrete = [false*ones(1,numel(opti.x)-num_of_iv) true*ones(1,num_of_iv)];
opts.expand=true; %When this option is true, it goes WAY faster!
opts.print_time=true;
if (strcmp(my_solver,'ipopt'))
    opts.ipopt.print_level=print_level; 
    opts.ipopt.print_frequency_iter=1e10;%1e10 %Big if you don't want to print all the iteratons
    opts.ipopt.linear_solver=linear_solver_name;
else
    opts.gurobi.verbose=true; 
end
% opts.error_on_fail=false;
opti.solver(my_solver,opts); %{"ipopt.hessian_approximation":"limited-memory"} 



all_wv=0:0.1:0.3;
all_wa=0:0.1:0.3;
all_wj=0:0.1:0.3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate samples inside the distribution
all_x={};
all_y={};
for wv=all_wv
    for wa=all_wa
        for wj=all_wj
            [wv;wa;wj]
            opti.set_value(weights,[wv;wa;wj]);
            sol = opti.solve();
            assert(strcmp(sol.stats.return_status,'Solve_Succeeded'))
            control_points=sol.value(sp.getCPsAsMatrix);
            all_x{end+1}=[wv;wa;wj];
            all_y{end+1}=control_points(:);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate samples outside the distribution
all_x_out_dist={};
all_y_out_dist={};
for wv=2*all_wv
    for wa=2*all_wa
        for wj=2*all_wj
            [wv;wa;wj]'
            opti.set_value(weights,[wv;wa;wj]);
            sol = opti.solve();
            assert(strcmp(sol.stats.return_status,'Solve_Succeeded'))
            control_points=sol.value(sp.getCPsAsMatrix);
            all_x_out_dist{end+1}=[wv;wa;wj];
            all_y_out_dist{end+1}=control_points(:);
        end
    end
end

%%%%%%Plotting
num_sol=numel(all_y);
dist_matrix=zeros(num_sol,num_sol);
for i=1:num_sol
    for j=1:num_sol
        tmp=vecnorm(all_y{i}-all_y{j});
        dist_matrix(i,j)=sum(tmp)/sp.num_cpoints;
    end
end
figure;
imagesc(dist_matrix); colorbar
%%%%%%%%%%%%%%%%%%

[Aineq,bineq]=getAbLinearConstraints(opti);
polyhedron.Aineq=Aineq;
polyhedron.bineq=bineq;


save('corridor.mat','all_x','all_y','all_x_out_dist','all_y_out_dist','polyhedron');

sol = opti.solve();

sp.updateCPsWithSolution(sol.value(sp.getCPsAsMatrix()));


%%


% fplot3(Pcurve(1,:),Pcurve(2,:),Pcurve(3,:),[0,1]); axis equal;

% ylim([-4,4]);zlim([-4,4]);

figure(1);
sp.plotPos3D(6)

plot3(P(1,:),P(2,:),P(3,:),'--','LineWidth',2)

view(48,38); axis equal

xlabel('x'); ylabel('y'); zlabel('z');

% sp.plotPosVelAccelJerk(v_max,a_max,j_max)
%%
% close all;
% figure; hold on;
% p1=[0;0;0];
% p2=[1;1;1];

%This gives the vertices of a polyhedron around the line p1-->p2
function [A, b, V]=getAbVerticesPolyhedronAroundP1P2(p1,p2, steps, samples_per_step, radius)


%     samples_per_step=5;
%     radius=0.5;
    all_points=[];

    for alpha=linspace(0,1,steps)
        point=alpha*p1 + (1-alpha)*p2;
        results=point + radius*uniformSampleInUnitBall(3,samples_per_step);
        all_points=[all_points results];
    end

    [k1,av1] = convhull(all_points');
    k_all_unique=unique(k1);
    
    V=all_points(:,k_all_unique); %V contains the vertices

    [A,b,Aeq,beq]=vert2lcon(V');

    assert(numel(Aeq)==0)
    assert(numel(beq)==0)

%     for tmp=k_all_unique'
%         tmp
%         P(:,tmp)
%         plotSphere(P(:,tmp), 0.03, 'r');
%     end


end

%This gives a polyhedron with 6 faces around the line p1-->p2
%Region is {x such that A1*x<=b1}
function [A, b]=getABgivenP1P2(p1,p2)

h=norm(p1-p2);

hside=0.4;
b_p1_a=[hside -hside 0]';
b_p1_b=[hside hside 0]';
b_p1_c=[-hside hside 0]';
b_p1_d=[-hside -hside 0]';

b_p2_a=[hside -hside h]';
b_p2_b=[hside hside h]';
b_p2_c=[-hside hside h]';
b_p2_d=[-hside -hside h]';

yaw=0.0;

zb=(p2-p1)/norm(p2-p1);
xb=cross([-sin(yaw) cos(yaw) 0]',zb); 
assert(norm(xb)>0)
xb=xb/norm(xb);
yb=cross(zb,xb);
w_R_b=[xb yb zb];

w_p1_a=w_R_b*b_p1_a+p1;
w_p1_b=w_R_b*b_p1_b+p1;
w_p1_c=w_R_b*b_p1_c+p1;
w_p1_d=w_R_b*b_p1_d+p1;

w_p2_a=w_R_b*b_p2_a+p1;
w_p2_b=w_R_b*b_p2_b+p1;
w_p2_c=w_R_b*b_p2_c+p1;
w_p2_d=w_R_b*b_p2_d+p1;

b_A=[1 0 0;
    0 1 0;
    0 0 1;
    -1 0 0;
    0 -1 0;
    0 0 -1];
A=[];

for i=1:size(b_A,1)
    A=[A;(w_R_b*b_A(i,:)')'];
end

b=[A(1,:)*w_p1_a;
   A(2,:)*w_p1_b;
   A(3,:)*w_p2_a;
   A(4,:)*w_p1_c;
   A(5,:)*w_p1_d;
   A(6,:)*w_p1_d;];
end


% for j=1:(sp.num_seg)
%     %Get the control points of the interval
% 
%     Q_matrix=cell2mat(sp.getCPs_XX_Pos_ofInterval("MINVO",j));
%     [k,av]=convhull(Q_matrix');
%     if(basis=="MINVO")
%         all_volumes.MINVO=[all_volumes.MINVO, av];
%     elseif(basis=="BEZIER")
%         all_volumes.BEZIER=[all_volumes.BEZIER, av];
%     end
%     
% end

% end

% %Region is {x such that A1*x<=b1}
% function [A1,b1]=getAb_Box(center,side)
% 
% A1=[1 0 0;
%     0 1 0;
%     0 0 1;
%     -1 0 0;
%     0 -1 0;
%     0 0 -1];
% 
% of_x=center(1);
% of_y=center(2);
% of_z=center(3);
% 
% b1=[side(1)/2.0+of_x;
%     side(2)/2.0+of_y;
%     side(3)/2.0+of_z;
%     side(1)/2.0-of_x;
%     side(2)/2.0-of_y;
%     side(3)/2.0-of_z];
% 
% end


% [allA, allb, p0, pf]=getCorridorConstraintsFromCurve();
% function [allA, allb, p0, pf]=getCorridorConstraintsFromCurve()
%     syms t
%     Pcurve= [ sin(t*3*pi)+2*sin(2*t*3*pi);
%               cos(t*3*pi)-2*cos(2*t*3*pi);
%              -sin(3*t*3*pi);];
% 
% 
%     p_last=subs(Pcurve,t,0.0);
%     allA={}; allb={};
%     delta=0.3;
%     last_tt=0;
%     for tt=delta:delta:3
% 
%         p1=double(p_last);
%         p2=double(subs(Pcurve,t,tt));
% 
%         [A b]=getABgivenP1P2(p1,p2);
%         allA{end+1}= A;
%         allb{end+1}=b;
%         p_last=p2;
%         last_tt=tt;
%     end
%     
%     p0=double(0.9*subs(Pcurve,t,0.0)+0.1*subs(Pcurve,t,delta));
%     pf=double(0.9*subs(Pcurve,t,last_tt)+0.1*subs(Pcurve,t,last_tt-delta));
% end


% polyhedron.Aeq=[]; %already included in (Aineq, bineq)
% polyhedron.beq=[];

% dataset.all_x=all_x;
% dataset.all_y=all_y;