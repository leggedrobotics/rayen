%This file solves the traj planning problem of generating a smooth
%trajectory passing through several 3d polytopes. 

%Note that in some cases (depending on the geometry of the convex corridor), the Bezier basis can be better than the MINVO
%basis (due to the fact that the Bezier basis includes the initial and
%final points --> better job in the overlapping areas between several
%polyhedra

close all; clc;clear;
doSetup();

dimension=3;

%%%So that random is repeatable
rng('default');
rng(1);
%%%%%%%%%%%%

t0=0.0;
tf=15.0;

if(dimension==2)
    P=3*[-0.1 1.0 2.5 3.5 5.5 6.8 7.5;
         3 1.0 1 0 4 4 0];
    radius=4.0;
    num_of_seg_per_region=1; 
    samples_per_step=5;
    N=15;
    use_quadratic=false;
else
    P=3*[0 1 2 3 4 3 2;
       0 1 1 2 4 4 4;
       0 1 1 1 4 1 0];
    radius=4*1.3;
    num_of_seg_per_region=3; 
    samples_per_step=5;
    N=5;
    use_quadratic=true;
end



allA={};
allb={};
allV={};

steps=2;



for i=1:(size(P,2)-1)
%     [A, b]=getABgivenP1P2(P(:,i),P(:,i+1));
    [A, b, V]=getAbVerticesPolyhedronAroundP1P2(P(:,i),P(:,i+1), steps, samples_per_step, radius);
    allA{end+1}=A;
    allb{end+1}=b;
    allV{end+1}=V;
end

p0=mean(allV{1},2); %  0.8*P(:,1) + 0.2*P(:,2);
pf=mean(allV{end},2); %0.2*P(:,end-1) + 0.8*P(:,end);

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

if(dimension==2)
    scatter(p0(1),p0(2),'filled','g')
    scatter(pf(1),pf(2),'filled','r')
end

if(dimension==3)
    zlim([min(P(3,:))-delta,max(P(3,:))+delta]);
    plotSphere(p0,0.2,'g')
    plotSphere(pf,0.2,'r')
end




%%


num_of_regions=size(allA,2);

%Note: IF YOU FIND THE ERROR "Matrix product with incompatible dimensions. Lhs is 3x1 and rhs is 3x3." when changing, this, the cause if the hand-coded "n_int_knots=15; " in computeMatrixForClampedUniformBSpline.m. Increase it.

basis="MINVO"; %MINVO OR B_SPLINE or BEZIER. This is the basis used for collision checking (in position, velocity, accel and jerk space), both in Matlab and in C++
linear_solver_name='ma27'; %mumps [default, comes when installing casadi], ma27, ma57, ma77, ma86, ma97 
my_solver='gurobi' %'ipopt' %'gurobi'
print_level=0; %From 0 (no verbose) to 12 (very verbose), default is 5

if (strcmp(my_solver,'gurobi'))
opti = casadi.Opti('conic');%'conic' I think you need to use 'conic' for gurobi
else
 opti = casadi.Opti();
end
deg_pos=3;
dim_pos=dimension;
num_seg =num_of_seg_per_region*num_of_regions;
sp=MyClampedUniformSpline(t0,tf,deg_pos, dim_pos, num_seg, opti);

guess=[p0(1)*ones(1,3) linspace(p0(1), pf(1), sp.num_cpoints-6) pf(1)*ones(1,3) 
       p0(2)*ones(1,3) linspace(p0(2), pf(2), sp.num_cpoints-6) pf(1)*ones(1,3)]

if(dimension==3)
    guess=[guess; p0(3)*ones(1,3) linspace(p0(3), pf(3), sp.num_cpoints-6) pf(1)*ones(1,3) ]
end

opti.set_initial(sp.getCPsAsMatrix,guess);

v0=zeros(dimension,1); a0=zeros(dimension,1);
vf=zeros(dimension,1); af=zeros(dimension,1);

v_max=4*ones(dimension,1);   a_max=6*ones(dimension,1);   j_max=50*ones(dimension,1);

linear_eq_const={};
%Initial conditions
linear_eq_const=[linear_eq_const   {sp.getPosT(t0)== p0}];
linear_eq_const=[linear_eq_const   {sp.getVelT(t0)== v0}];
linear_eq_const=[linear_eq_const   {sp.getAccelT(t0)== a0}];

%Final conditions
% opti.subject_to( sp.getPosT(tf)== pf);
linear_eq_const=[linear_eq_const   {sp.getVelT(tf)==vf}];
linear_eq_const=[linear_eq_const   {sp.getAccelT(tf)==af}];


dyn_lim_const={};
dyn_lim_const=[dyn_lim_const sp.getMaxVelConstraints(basis, v_max, use_quadratic)];      %Max vel constraints (position)
dyn_lim_const=[dyn_lim_const sp.getMaxAccelConstraints(basis, a_max, use_quadratic)];    %Max accel constraints (position)
if(deg_pos>=3)
    dyn_lim_const=[dyn_lim_const sp.getMaxJerkConstraints(basis, j_max, use_quadratic)];     %Max jerk constraints (position)
end

linear_ineq_const={};
quadratic_const={};

if(use_quadratic)
    quadratic_const = dyn_lim_const;
else
    linear_ineq_const = dyn_lim_const;
end

%Corridor constraints
for j=1:(sp.num_seg)
    
     Q=sp.getCPs_XX_Pos_ofInterval(basis, j);%Get the control points of the interval
    
    %%%%%To force the interval allocation:
    ip=ceil(j/num_of_seg_per_region); %ip is the index of the polyhedron
    
    for kk=1:size(Q,2)
            linear_ineq_const=[linear_ineq_const {allA{ip}*Q{kk}<=allb{ip}}];   %Each segment must be in each corridor
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%% EXTRACT THE CONSTRAINTS
opti.subject_to(); %delete any constraints in the model
opti.subject_to(linear_eq_const);
opti.subject_to(linear_ineq_const);
[A1,b1]=getAbLinearConstraints(opti);

opti.subject_to(); %delete any constraints in the model
opti.subject_to(quadratic_const);
[all_P, all_q, all_r]=getAllPqrQuadraticConstraints(opti);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opti.subject_to(); %delete any constraints in the model
opti.subject_to(linear_eq_const);
opti.subject_to(linear_ineq_const);
opti.subject_to(quadratic_const);

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
opts.expand=true; %When this option is true, it goes WAY faster!
opts.print_time=true;
if (strcmp(my_solver,'ipopt'))
    opts.ipopt.print_level=print_level; 
    opts.ipopt.print_frequency_iter=1e10;%1e10 %Big if you don't want to print all the iteratons
    opts.ipopt.linear_solver=linear_solver_name;
else
%     opts.gurobi.verbose=true; 
end
% opts.error_on_fail=false;
opti.solver(my_solver,opts); %{"ipopt.hessian_approximation":"limited-memory"} 



a=0.001;
b=0.3;
all_wv=[a + (b-a).*rand(N,1)]'; 
all_wa=[a + (b-a).*rand(N,1)]'; 
all_wj=[a + (b-a).*rand(N,1)]'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate samples inside the distribution

[all_x, all_y, all_Pobj, all_qobj, all_robj, all_costs, all_times_s]=solveProblemForDifferentGamma(all_wv, all_wa, all_wj, weights, my_solver, opti, sp, cost);
factor=5;
[all_x_out_dist, all_y_out_dist, all_Pobj_out_dist, all_qobj_out_dist, all_robj_out_dist, all_costs_out_dist, all_times_s_out_dist]=solveProblemForDifferentGamma(factor*all_wv, factor*all_wa, factor*all_wj, weights, my_solver, opti, sp, cost);

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
imagesc(dist_matrix); colorbar; axis equal
%%%%%%%%%%%%%%%%%%

save(['corridor_dim',num2str(dimension),'.mat'],'all_x','all_y','all_Pobj','all_qobj','all_robj','all_costs','all_times_s', ...
     'all_x_out_dist','all_y_out_dist','all_Pobj_out_dist','all_qobj_out_dist','all_robj_out_dist','all_costs_out_dist','all_times_s_out_dist', ...
     'A1','b1','all_P','all_q','all_r');

sol = opti.solve();

%%
%%%%%
% control_points=sp.getCPsAsMatrix();
% variables=;
% 

%%%%%
%%
sp.updateCPsWithSolution(sol.value(sp.getCPsAsMatrix()));


%%


% fplot3(Pcurve(1,:),Pcurve(2,:),Pcurve(3,:),[0,1]); axis equal;

% ylim([-4,4]);zlim([-4,4]);

figure(1);
xlabel('x (m)'); ylabel('y (m)');
if(dimension==2)
    sp.plotPos2D(6)
    plot(P(1,:),P(2,:),'--','LineWidth',2)
else
    sp.plotPos3D(6)
    zlabel('z (m)');
    view(48,38);
    % plot3(P(1,:),P(2,:),P(3,:),'--','LineWidth',2)
end



 axis equal

 

% sp.plotPosVelAccelJerk(v_max,a_max,j_max)
%%
% close all;
% figure; hold on;
% p1=[0;0;0];
% p2=[1;1;1];


function [all_x, all_y, all_Pobj, all_qobj, all_robj, all_costs, all_times_s]=solveProblemForDifferentGamma(all_wv, all_wa, all_wj, weights, my_solver, opti, sp, cost)
    all_x={};
    all_y={};
    all_Pobj={};
    all_qobj={};
    all_robj={};
    all_costs={};
    all_times_s={};
    i=0
    total_iterations=numel(all_wv)*numel(all_wa)*numel(all_wj);
    for wv=all_wv
        for wa=all_wa
            for wj=all_wj
                i=i+1;
                i/total_iterations
                weights_value=[wv;wa;wj];
                opti.set_value(weights,weights_value);
                display("Going to solve")
                sol = opti.solve();
                display("Solved")
                checkSolverSucceeded(sol, my_solver)
                control_points=sol.value(sp.getCPsAsVector());
                all_x{end+1}=weights_value;
                all_y{end+1}=control_points(:);
                cost_substituted=casadi.substitute(cost,weights,weights_value);
                [P,q,r]=getPandqandrOfQuadraticExpressionCasadi(cost_substituted, sp.getCPsAsVector());
                all_Pobj{end+1}=P;   
                all_qobj{end+1}=q;
                all_robj{end+1}=r;
                all_costs{end+1}=sol.value(cost);
                all_times_s{end+1}=opti.stats().t_wall_solver; %This is the time the solver takes to solve the problem, see (for Gurobi) https://github.com/casadi/casadi/blob/e5d6977d621e3b7a0cd0b2e24cdd2b73a6c3a8fe/casadi/interfaces/gurobi/gurobi_interface.cpp#L447
            end
        end
    end
end



function checkSolverSucceeded(sol, my_solver)
    if (strcmp(my_solver,'ipopt'))
        assert(strcmp(sol.stats.return_status,'Solve_Succeeded'))
    end
    if (strcmp(my_solver,'gurobi'))
        assert(strcmp(sol.stats.return_status,'OPTIMAL'), sol.stats.return_status)
    end
end

%This gives the vertices of a polyhedron around the line p1-->p2
function [A, b, V]=getAbVerticesPolyhedronAroundP1P2(p1,p2, steps, samples_per_step, radius)


    dimension=size(p1,1);

%     samples_per_step=5;
%     radius=0.5;
    all_points=[];

    for alpha=linspace(0,1,steps)
        point=alpha*p1 + (1-alpha)*p2;
        results=point + radius*uniformSampleInUnitBall(dimension,samples_per_step);
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