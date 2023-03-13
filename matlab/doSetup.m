function doSetup()
    close all; clc;clear;

    set(0,'DefaultFigureWindowStyle','docked') %'normal' 'docked'
    set(0,'defaulttextInterpreter','latex');  set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
    set(0,'defaultfigurecolor',[1 1 1])
    
    import casadi.*
    addpath(genpath('./../deep_panther/panther/matlab'))
    addpath(genpath('./../deep_panther/submodules/minvo'))
    addpath(genpath('./utils'))

end