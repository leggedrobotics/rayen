close all; clc;clear;
doSetup();


dimension=2;
[allA, allb, allV, p0, t0,tf,deg_pos, num_seg, num_of_seg_per_region, use_quadratic]=getCorridorAndParamsSpline(dimension);
load('./../results/results_test_in_dist_dataset2d_RAYEN_weight_soft_cost_0.0.mat')

opti = casadi.Opti('conic');%'conic' I think you need to use 'conic' for gurobi
sp=MyClampedUniformSpline(t0,tf,deg_pos, dimension, num_seg, opti);

all_x=squeeze(all_x);
all_y=squeeze(all_y);
all_y_predicted=squeeze(all_y_predicted);
% figure;
hold on;
for i=1:10:500
    i
    c_points=all_y_predicted(i,:)';
    c_points=reshape(c_points,dimension,[]); 
    sp.updateCPsWithSolution(c_points);
    xlabel('x (m)'); ylabel('y (m)');
    if(dimension==2)
        sp.plotPos2D(1)
    else
        sp.plotPos3D(6)
        zlabel('z (m)');
        view(48,38);
        % plot3(P(1,:),P(2,:),P(3,:),'--','LineWidth',2)
    end

end