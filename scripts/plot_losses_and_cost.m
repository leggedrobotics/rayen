close all; clear; clc;

set(0,'DefaultFigureWindowStyle','normal') %'normal' 'docked'
set(0,'defaulttextInterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
set(0,'defaultfigurecolor',[1 1 1])

addpath(genpath('./../submodules/minvo/'))
addpath(genpath('./../submodules/export_fig/'))
addpath(genpath('./../matlab/utils'))

my_table=readtable('./results/merged.csv');

hasMatch = ~cellfun('isempty', regexp(my_table.method, 'dataset2d', 'once')) ;
results2d=my_table(hasMatch, :);

hasMatch = ~cellfun('isempty', regexp(my_table.method, 'dataset3d', 'once')) ;
results3d=my_table(hasMatch, :);


position=[1000        1117         560         205];
figure;
tcl = tiledlayout(1,2);
nexttile(tcl); hold on;
[circle_plots, names ]=plotTimevsCost(results2d,"InDist", "\textbf{Inside dist.}");
hL = legend(circle_plots,names{:},'Location','eastoutside'	);
hL.Layout.Tile = 'East';
ylim([0.9,1.15])

nexttile(tcl); hold on;
plotTimevsCost(results2d,"OutDist", "\textbf{Outside dist.}")
ylim([0.9,1.15])
title(tcl,'\textbf{Optimization 1}','interpreter','latex')
set(gcf,'Position',position)
export_fig time_loss_opt1.png -m2.5

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
export_fig time_loss_opt2.png -m2.5


%% Plot Model Complexity

position=[1000        1117         560         105];

figure; hold on;
plotModelComplexity(results2d, position)
export_fig model_complexity_opt1.png -m2.5
ylim([0,15*10^4])

figure; hold on;
plotModelComplexity(results3d, position)
ylim([0,15*10^4])
export_fig model_complexity_opt2.png -m2.5


function plotModelComplexity(results, position)

hasMatch = ~cellfun('isempty', regexp(results.method, '10', 'once')) ;
results=results(~hasMatch, :);

hasMatch = ~cellfun('isempty', regexp(results.method, 'Optimization', 'once')) ;
results=results(~hasMatch, :);

hasMatch = ~cellfun('isempty', regexp(results.method, 'walker_2', 'once'));
results=results(~hasMatch, :);



for i=1:numel(results.method)
    
    name=results.method(i);
    name=name{1};
    name=strrep(name,'dataset2d_','');
    name=strrep(name,'dataset3d_','');
    name = extractBefore(name,'_weight_');
    name=strrep(name,'walker_1','RAYEN');
    name=['\textbf{',name,'}'];
    results.method{i}=name;  

end

X = categorical(results.method');
X = reordercats(X,results.method');
bar(X,results.num_trainable_params)

% set(gca, 'XTickLabel', results.method');
ylabel('Params.')
set(gcf,'Position',position)


end


%% 

function [circle_plots, names]=plotTimevsCost(results, type, my_title)

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
        if(contains(name,"walker_2"))
            continue
        end
        name=strrep(name,'walker_1','RAYEN');
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
    
    
    title(my_title)


end

% set(gca, 'XScale', 'log')
% set(gca, 'YScale', 'log')