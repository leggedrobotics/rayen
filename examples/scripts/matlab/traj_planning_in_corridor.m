%This file solves the traj planning problem of generating a smooth
%trajectory passing through several 3d polytopes. 

%Note that in some cases (depending on the geometry of the convex corridor), the Bezier basis can be better than the MINVO
%basis (due to the fact that the Bezier basis includes the initial and
%final points --> better job in the overlapping areas between several
%polyhedra

close all; clc;clear;
doSetup();

dimension=3;


[allA, allb, allV, p0, t0,tf,deg_pos, num_seg, num_of_seg_per_region, use_quadratic]=getCorridorAndParamsSpline(dimension)

% 
% 
% 
% scatter(points(1,:),points(2,:))
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
dim_pos=dimension;
sp=MyClampedUniformSpline(t0,tf,deg_pos, dim_pos, num_seg, opti);

% guess=[p0(1)*ones(1,3) linspace(p0(1), pf(1), sp.num_cpoints-6) pf(1)*ones(1,3) 
%        p0(2)*ones(1,3) linspace(p0(2), pf(2), sp.num_cpoints-6) pf(1)*ones(1,3)]
% 
% if(dimension==3)
%     guess=[guess; p0(3)*ones(1,3) linspace(p0(3), pf(3), sp.num_cpoints-6) pf(1)*ones(1,3) ]
% end
% 
% opti.set_initial(sp.getCPsAsMatrix,guess);

v_max=4*ones(dimension,1);   a_max=6*ones(dimension,1);   j_max=50*ones(dimension,1);

zero = zeros(dimension,1);

linear_eq_const={};

%%%Initial and final conditions

%Position
linear_eq_const=[linear_eq_const   {sp.getPosT(t0) == p0}];

%Velocity 
linear_eq_const=[linear_eq_const   {sp.getVelT(t0) == zero}];
linear_eq_const=[linear_eq_const   {sp.getVelT(tf) == zero}];

%Accel
if(deg_pos>=3)
    linear_eq_const=[linear_eq_const   {sp.getAccelT(t0) == zero}];
    linear_eq_const=[linear_eq_const   {sp.getAccelT(tf) == zero}];
end


%%%Dynamic limits and final conditions
dyn_lim_const={};
dyn_lim_const=[dyn_lim_const sp.getMaxVelConstraints(basis, v_max, use_quadratic)];          %Max vel constraints (position)
dyn_lim_const=[dyn_lim_const sp.getMaxAccelConstraints(basis, a_max, use_quadratic)];        %Max accel constraints (position)
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
%     if(j==1)
%         ip=1;  %ip is the index of the polyhedron
%     else
%         ip=ceil((j+1)/num_of_seg_per_region); 
%     end
    ip=ceil((j)/num_of_seg_per_region); 
    [j, ip]
    
    for kk=1:size(Q,2)
            linear_ineq_const=[linear_ineq_const {allA{ip}*Q{kk}<=allb{ip}}];   %Each segment must be in each corridor
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%% EXTRACT THE CONSTRAINTS
opti.subject_to(); %delete any constraints in the model
opti.subject_to(linear_ineq_const);
opti.subject_to(linear_eq_const);
[A1,b1, A2, b2]=getAbLinearConstraints(opti);

opti.subject_to(); %delete any constraints in the model
opti.subject_to(quadratic_const);
[all_P, all_q, all_r]=getAllPqrQuadraticConstraints(opti);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opti.subject_to(); %delete any constraints in the model
opti.subject_to(linear_eq_const);
opti.subject_to(linear_ineq_const);
opti.subject_to(quadratic_const);

num_variables=numel(opti.x);
num_linear_ineq_const=countNumOfConstraints(linear_ineq_const);
num_linear_eq_const=countNumOfConstraints(linear_eq_const);
num_quadratic_const=countNumOfConstraints(quadratic_const);


%%


%%

weights=opti.parameter(3,1);
pf=opti.parameter(dimension,1);

%%%%%%%%%%%%%%%%%% COST
vel_cost=sp.getVelCost();
accel_cost=sp.getAccelCost();
jerk_cost=sp.getControlCost();
final_pos_cost=(sp.getPosT(tf)- pf)'*(sp.getPosT(tf)- pf);

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
%      opts.gurobi.verbose=true; 
end
% opts.error_on_fail=false;
opti.solver(my_solver,opts); %{"ipopt.hessian_approximation":"limited-memory"} 



rng('shuffle')

N_inside=1728;
N_outside=512;

% N_inside=5;
% N_outside=5;

all_gammas=randInInterval(0.0,1.0,[3, N_inside]);
all_pf=cprnd(N_inside,allA{end},allb{end})';

[all_x, all_y, all_Pobj, all_qobj, all_robj, all_costs, all_times_s]=solveProblemForDifferentGamma(all_gammas, all_pf, weights, pf, my_solver, opti, sp, cost);
all_gammas=randInInterval(1.0,2.0,[3, N_outside]);
[all_x_out_dist, all_y_out_dist, all_Pobj_out_dist, all_qobj_out_dist, all_robj_out_dist, all_costs_out_dist, all_times_s_out_dist]=solveProblemForDifferentGamma(all_gammas, all_pf, weights, pf, my_solver, opti, sp, cost);

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
     'A1','b1','A2','b2','all_P','all_q','all_r');

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

figure(1);
xlabel('x (m)'); ylabel('y (m)');
if(dimension==2)
    sp.plotPos2D(6)
%     plot(P(1,:),P(2,:),'--','LineWidth',2)
else
    sp.plotPos3D(6)
    zlabel('z (m)');
    view(48,38);
    % plot3(P(1,:),P(2,:),P(3,:),'--','LineWidth',2)
end

axis equal


% sp.plotPosVelAccelJerk(v_max,a_max,j_max)
%%

function [all_x, all_y, all_Pobj, all_qobj, all_robj, all_costs, all_times_s]=solveProblemForDifferentGamma(all_gammas, all_pf, weights, pf, my_solver, opti, sp, cost)
    all_x={};
    all_y={};
    all_Pobj={};
    all_qobj={};
    all_robj={};
    all_costs={};
    all_times_s={};
    num_gammas=size(all_gammas,2);
    for i=1:num_gammas
        i/num_gammas
        weights_value=all_gammas(:,i)
        pf_value=all_pf(:,i)
        opti.set_value(weights,weights_value);
        opti.set_value(pf,pf_value);
        display("Going to solve")
        sol = opti.solve();
        display("Solved")
        checkSolverSucceeded(sol, my_solver)
        control_points=sol.value(sp.getCPsAsVector());
        all_x{end+1}=[weights_value; pf_value];
        all_y{end+1}=control_points(:);
        cost_substituted=cost;
        cost_substituted=casadi.substitute(cost_substituted,weights,weights_value);
        cost_substituted=casadi.substitute(cost_substituted,pf,pf_value);
        [P,q,r]=getPandqandrOfQuadraticExpressionCasadi(cost_substituted, sp.getCPsAsVector());
        all_Pobj{end+1}=P;   
        all_qobj{end+1}=q;
        all_robj{end+1}=r;
        all_costs{end+1}=sol.value(cost);
        all_times_s{end+1}=opti.stats().t_wall_solver; %This is the time the solver takes to solve the problem, see (for Gurobi) https://github.com/casadi/casadi/blob/e5d6977d621e3b7a0cd0b2e24cdd2b73a6c3a8fe/casadi/interfaces/gurobi/gurobi_interface.cpp#L447

       %%%% If you want to plot this specific trajectory
%         opti_tmp=opti.copy;
%         sp_novale=MyClampedUniformSpline(sp.t0,sp.tf,sp.p, sp.dim, sp.num_seg, opti_tmp);
%         sp_novale.updateCPsWithSolution(sol.value(sp.getCPsAsMatrix()));
%         xlabel('x (m)'); ylabel('y (m)');
%         if(sp.dim==2)
%             sp_novale.plotPos2D(2)
%         else
%             sp_novale.plotPos3D(2)
%             zlabel('z (m)');
%             view(48,38);
%         end
        %%%%%%%%%%%%%%%%%%%%%%%%
            
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

function num_of_constraints=countNumOfConstraints(constraints)

num_of_constraints=0;
for i=1:numel(constraints)
    num_of_constraints= num_of_constraints + numel(constraints{i});
end

end
