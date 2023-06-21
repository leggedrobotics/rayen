close all; clc;clear;

set(0,'DefaultFigureWindowStyle','normal') %'normal' 'docked'
set(0,'defaulttextInterpreter','latex');  set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
set(0,'defaultfigurecolor',[1 1 1])

import casadi.*
addpath(genpath('./matlab/'))


position=[675         165        1041         797];
doPlot(2); ylim([-5,17])
set(gcf,'Position',[675         165        1041/1.2         797/1.2])
export_fig trajectories2d.png -m2.5


doPlot(3)
view(-90,90); set(gcf,'Position',position)
export_fig trajectories3d_top.png -m2.5
view(-59,23); set(gcf,'Position',position)
export_fig trajectories3d_side.png -m2.5

function doPlot(dimension)

    [allA, allb, allV, p0, t0,tf,deg_pos, num_seg, num_of_seg_per_region, use_quadratic]=getCorridorAndParamsSpline(dimension);
    load(['./results/results_test_in_dist_dataset',num2str(dimension),'d_RAYEN_weight_soft_cost_0.0.mat'])
    
    opti = casadi.Opti('conic');%'conic' I think you need to use 'conic' for gurobi
    sp=MyClampedUniformSpline(t0,tf,deg_pos, dimension, num_seg, opti);
    
    all_x=squeeze(all_x);
    all_y=squeeze(all_y);
    all_y_predicted=squeeze(all_y_predicted);
    % figure;
    hold on;
    
    num_samples=100;
    indexes_all_samples=randperm(size(all_y_predicted,1),num_samples); %https://ch.mathworks.com/matlabcentral/answers/71181-how-to-generate-non-repeating-random-numbers-from-1-to-49#answer_81589
    j=0;
    for i=indexes_all_samples
        j/num_samples
        c_points=all_y_predicted(i,:)';
        c_points=reshape(c_points,dimension,[]); 
        sp.updateCPsWithSolution(c_points);
        xlabel('x (m)'); ylabel('y (m)');
        if(dimension==2)
            sp.plotPos2D(1,'r')
        else
            sp.plotPos3D(1,'r')
            zlabel('z (m)');
            view(48,38);
            % plot3(P(1,:),P(2,:),P(3,:),'--','LineWidth',2)
        end
       j=j+1; 
    end

end

