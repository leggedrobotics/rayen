close all; clear; clc;

set(0,'DefaultFigureWindowStyle','normal') %'normal' 'docked'
set(0,'defaulttextInterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
set(0,'defaultfigurecolor',[1 1 1])

addpath(genpath('./../submodules/minvo/'))
addpath(genpath('./../submodules/export_fig/'))
addpath(genpath('./utils'))


lin=readtable('./results/times_lin.csv');
qp=readtable('./results/times_qp.csv');
soc=readtable('./results/times_soc.csv');
lmi=readtable('./results/times_lmi.csv');

default_colors = colororder(); 


linewidth=1.2;
w=297;
w_soc=466;
position=[228   722   w   427];
position_soc=[228   722   w_soc   427];
tmp=[w w w w_soc]
tmp/sum(tmp) %This is for latex
% %Linear CONSTRAINTS
figure; hold on;  legend;
num_color=1;
for i=sort(unique(lin.r_A1)','descend')
    if(i==1 || i==3000)%|| r_M==200
        continue
    end
    indexes=(lin.r_A1==i);
    plot(lin.k(indexes), 1e3*lin.Time(indexes),'-o','DisplayName',['$r=',num2str(i),'$'], 'LineWidth',linewidth, 'MarkerFaceColor',default_colors(num_color,:),'MarkerEdgeColor',default_colors(num_color,:))
    xlabel('$k$'); ylabel('Time (ms)')
    title("\textbf{Linear Constraints}")
    legend('Location','northwest')
    num_color=num_color+1;
end
set(gcf, 'Position', position); 
export_fig time_linear.png -m2.5

% 
% %QP CONSTRAINTS
figure; hold on;  legend;
num_color=1;
for i=sort(unique(qp.eta)','descend')
    indexes=(qp.eta==i);
    plot(qp.k(indexes), 1e3*qp.Time(indexes),'-o','DisplayName',['$\eta=',num2str(i),'$'], 'LineWidth',linewidth, 'MarkerFaceColor',default_colors(num_color,:),'MarkerEdgeColor',default_colors(num_color,:))
    xlabel('$k$'); ylabel('Time (ms)')
    title("\textbf{QP Constraints}")
     legend('Location','northwest')
     num_color=num_color+1;
end
set(gcf, 'Position', position); 
export_fig time_qp.png -m2.5

%SOC CONSTRAINTS
% tiledlayout(2,2); 
figure; hold on;
for mu=sort(unique(soc.mu)','ascend')
    if(mu==1 || mu==200 || mu==400 )%|| r_M==200
        continue
    end
    nexttile; hold on;  legend;
    num_color=1;
    for r_M=sort(unique(soc.r_M)','descend')
        if(r_M==1 )%|| r_M==200
            continue
        end
        indexes=(soc.r_M==r_M & soc.mu==mu); %r_{\emph{\textbf M}}
        plot(soc.k(indexes), 1e3*soc.Time(indexes),'-o','DisplayName',['$r=',num2str(r_M),'$'], 'LineWidth',linewidth, 'MarkerFaceColor',default_colors(num_color,:),'MarkerEdgeColor',default_colors(num_color,:))
        xlabel('$k$'); ylabel('Time (ms)')
        title(['$\mu=',num2str(mu),'$'])
        legend('Location','northwest')
        num_color=num_color+1;
    end 
end
sgtitle('\textbf{SOC Constraints}', 'Interpreter','latex')
set(gcf, 'Position', position_soc); 
export_fig time_soc.png -m2.5

% %LMI CONSTRAINTS
num_color=1;
figure; hold on; legend;
for i=sort(unique(lmi.r_F)','descend')
    indexes=(lmi.r_F==i);
    plot(lmi.k(indexes), 1e3*lmi.Time(indexes),'-o','DisplayName',['$r=',num2str(i),'$'], 'LineWidth',linewidth, 'MarkerFaceColor',default_colors(num_color,:),'MarkerEdgeColor',default_colors(num_color,:))
    xlabel('$k$'); ylabel('Time (ms)')
    title("\textbf{LMI Constraints}")
    legend('Location','northwest')
    num_color=num_color+1;
end
set(gcf, 'Position', position); 
export_fig time_lmi.png -m2.5



