close all; clear; clc;

set(0,'DefaultFigureWindowStyle','normal') %'normal' 'docked'
set(0,'defaulttextInterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
set(0,'defaultfigurecolor',[1 1 1])

addpath(genpath('./matlab/submodules/minvo/'))
addpath(genpath('./matlab/submodules/export_fig/'))
addpath(genpath('./matlab/utils'))

my_table=readtable('./results/merged.csv');

%Remove RAYEN_old method
hasMatch = ~cellfun('isempty', regexp(my_table.method, 'RAYEN_old', 'once')) ;
my_table(hasMatch, :)=[];

%Change the name Optimization with Gurobi
hasMatch = find(~cellfun('isempty', regexp(my_table.method, 'dataset2d_Optimization', 'once')),1) ;
my_table(hasMatch, 1).method={'dataset2d_Gurobi'};

hasMatch = find(~cellfun('isempty', regexp(my_table.method, 'dataset3d_Optimization', 'once')),1) ;
my_table(hasMatch, 1).method={'dataset3d_Gurobi'};



hasMatch = ~cellfun('isempty', regexp(my_table.method, 'dataset2d', 'once')) ;
results2d=my_table(hasMatch, :);
results2d = moveToEndOfTable(results2d, 'dataset2d_DC3_weight_soft_cost_5000.0');
results2d = moveToEndOfTable(results2d, 'dataset2d_RAYEN_weight_soft_cost_0.0');


hasMatch = ~cellfun('isempty', regexp(my_table.method, 'dataset3d', 'once')) ;
results3d=my_table(hasMatch, :);
results3d = moveToEndOfTable(results3d, 'dataset3d_RAYEN_weight_soft_cost_0.0'); %move RAYEN to the last position


%%%%%%%%%%%%%%%%%%%%%%%% FIGURES WITHOUT LINE BREAK
position=[1000        1117         560         305];

%%% DATASET 2D
figure;
tcl = tiledlayout(1,2);
nexttile(tcl); hold on;
[circle_plots, names ]=plotTimevsCost(results2d,"InDist", "\textbf{Inside dist.}");
hL = legend(circle_plots,names{:},'Location','eastoutside'	);
hL.Layout.Tile = 'East';

nexttile(tcl); hold on;
plotTimevsCost(results2d,"OutDist", "\textbf{Outside dist.}")
title(tcl,'\textbf{Optimization 1}','interpreter','latex')
set(gcf,'Position',position)
export_fig time_loss_opt1_no_break.png -m2.5

%%% DATASET 3D
figure;
tcl = tiledlayout(1,2);
nexttile(tcl); hold on;
[circle_plots, names ]=plotTimevsCost(results3d,"InDist", "\textbf{Inside dist.}");
hL = legend(circle_plots,names{:},'Location','eastoutside'	);
hL.Layout.Tile = 'East';
nexttile(tcl); hold on;
plotTimevsCost(results3d,"OutDist", "\textbf{Outside dist.}")
title(tcl,'\textbf{Optimization 2}','interpreter','latex')

set(gcf,'Position',position)
export_fig time_loss_opt2_no_break.png -m2.5

%%%%%%%%%%%%%%%%%%%%%%%% FIGURES WITH LINE BREAK
%%% DATASET 2D
figure; subplot(1,2,1); hold on;
plotTimevsCost(results2d,"InDist", "\textbf{Inside dist.}");
min_y=0.95; max_y=11.85;ylim([min_y,max_y]);yticks(1.0:0.05:max_y); 
breakyaxis([1.16,11.8])

subplot(1,2,2); hold on;
plotTimevsCost(results2d,"OutDist", "\textbf{Outside dist.}")
min_y=0.95; max_y=4.7;ylim([min_y,max_y]);yticks(1.0:0.1:max_y); 
breakyaxis([1.45,4.56])

set(gcf,'Position',position)
export_fig time_loss_opt1_break.png -m2.5

%%% DATASET 3D
figure; subplot(1,2,1); hold on;
plotTimevsCost(results3d,"InDist", "\textbf{Inside dist.}");
min_y=0.95; max_y=8.0;ylim([min_y,max_y]);yticks(1.0:0.15:max_y); 
breakyaxis([2.5,7.8])

subplot(1,2,2); hold on;
plotTimevsCost(results3d,"OutDist", "\textbf{Outside dist.}")
min_y=0.95; max_y=10.4;ylim([min_y,max_y]);yticks(1.0:0.15:max_y); 
breakyaxis([2.6,10.2])

set(gcf,'Position',position)
export_fig time_loss_opt2_break.png -m2.5


%% Plot Model Complexity

position=[1000        1117         560         205];

figure; hold on;
plotModelComplexity(results2d, position, ["UU", "UP", "PP", "DC3", "Bar", "RAYEN"])
% export_fig model_complexity_opt1.png -m2.5
% ylim([0,15*10^4])

figure; hold on;
plotModelComplexity(results3d, position, ["UU", "UP", "PP", "DC3", "RAYEN"])
ylim([0,15*10^4])
% export_fig model_complexity_opt2.png -m2.5


function plotModelComplexity(results, position, names)

indexes=[];
for n=names
    n
    tmp=find(~cellfun('isempty', regexp(results.method, n, 'once')),1)
    indexes=[indexes, tmp]
end

X = categorical(names);
% X = reordercats(X,results.method');
bar(X,results.num_trainable_params(indexes))
% set(gca,'YScale','log')
% set(gca, 'XTickLabel', results.method');
ylabel('Params.')
set(gcf,'Position',position)


end


%% 

function my_table=moveToEndOfTable(my_table, my_string)
    match=~cellfun('isempty', regexp(my_table.method, my_string, 'once'));
    tmp=my_table(match, :); my_table(match, :)=[]; my_table=[my_table; tmp];
end

function [circle_plots, names]=plotTimevsCost(results, type, my_title)
%     figure;
    circle_plots=[];
    names={};
    has_bar=false;
    for i=1:numel(results.method)
        if(type=="InDist")
            t = results.x_InDist_Time_us(i);
            n_loss = results.x_InDist_N_loss(i);
            violation = results.x_InDist_Violation(i);
        else
            t = results.x_OutDist_Time_us(i);
            n_loss = results.x_OutDist_N_loss(i);
            violation = results.x_OutDist_Violation(i);
        end

        name=results.method(i);
        name=name{1};
        name=strrep(name,'dataset2d_','');
        name=strrep(name,'dataset3d_','');
        name=strrep(name,'_weight_soft_cost_',', $\omega=$ ');
        if(contains(name,"Bar"))
            has_bar=true;
        end
        if(contains(name,"RAYEN") || contains(name,"PP") || contains(name,"Bar"))
            name = extractBefore(name,', $\omega=$ ');  %These algorithms don't use omega
        end
    
        if(violation>1e-6)
            continue
        end
    
        tmp=scatter(t,n_loss,80.0,'o',"filled",'MarkerEdgeColor','k', 'LineWidth',1.0);
        circle_plots=[circle_plots tmp(1)];
        names{end+1}=name;
    end
    
    xlabel('Time ($\mu$s)')
    ylabel('N. Loss')
    yline(1.0,'--')
    color_bar=[0.5,0.5,0.5];
    color_rayen=[0,1,0];
    if(has_bar)
        tmp=numel(circle_plots)-1;
    else
        tmp=numel(circle_plots);
    end
    my_colormap=maxdistcolor(tmp,@sRGB_to_OKLab, 'exc',[color_bar; 1,1,1; color_rayen], 'Lmin',0.6, 'Lmax',1.0);%, 'inc',[1,0,0; 0,1,0]
    if(has_bar)
        my_colormap=[color_bar ;my_colormap];
    end
    my_colormap(end,:) = color_rayen;
    set(gca,'colororder',my_colormap);
    
%     set(gca, 'YScale', 'log')




    title(my_title)
      


end

% set(gca, 'XScale', 'log')
% set(gca, 'YScale', 'log')