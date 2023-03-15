function doSetup()

    set(0,'DefaultFigureWindowStyle','docked') %'normal' 'docked'
    set(0,'defaulttextInterpreter','latex');  set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
    set(0,'defaultfigurecolor',[1 1 1])
    
    import casadi.*
    addpath(genpath('./../submodules/minvo/'))
    addpath(genpath('./../submodules/export_fig/'))
    addpath(genpath('./utils'))

end