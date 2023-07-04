% --------------------------------------------------------------------------
% Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
% See LICENSE file for the license information
% -------------------------------------------------------------------------- 

addpath(genpath('./../submodules/minvo/'))
addpath(genpath('./../submodules/export_fig/'))
addpath(genpath('./utils'))

clear; clc;
t0=2.1;
tf=3.9;
import casadi.*

opti = casadi.Opti();%'conic' I think you need to use 'conic' for gurobi
deg_pos=3;
dim_pos=1;
num_seg =1;
sp=MyClampedUniformSpline(t0,tf,deg_pos, dim_pos, num_seg, opti);
cpoints=rand(dim_pos,sp.num_cpoints);
cpoints_cell_array=mat2cell(cpoints,dim_pos,ones(1,sp.num_cpoints));

sp.setCPoints(cpoints_cell_array)

vel_cost=sp.getVelCost();
convertMX2Matlab(vel_cost)
sp.getApproxVelCost(300)

accel_cost=sp.getAccelCost();
convertMX2Matlab(accel_cost)
sp.getApproxAccelCost(600)
%%
% opti = casadi.Opti();%'conic' I think you need to use 'conic' for gurobi
% p=opti.variable(1,4);
% p_value=[4 3 6 9];
% 
% t=MX.sym('t')
% pt=p*[t^3;t^2;t;1]
% 
% deg_spline=3;
% 
% vt=gradient(pt,t);
% at=gradient(vt,t);
% 
% vt_squared=vt*vt; %[at^5 + bt^4 +ct^3 +dt^2 +...]
% 
% v_squared=getCoeffPolyCasadi(vt_squared, t, 2*deg_spline); %[a b c d ...]
% 
% 
% tfinal=1.0;
% t0=0.0
% 
% tmp=[numel(v_squared):-1:1];
% result=sum(v_squared.*((tfinal.^tmp)./tmp)) - sum(v_squared.*((t0.^tmp)./tmp))
% result=convertMX2Matlab(substitute(result,p,p_value))
% 
% v=polyder(p_value);
% v_squared=conv(v,v);
% diff(polyval(polyint(v_squared),[t0 tfinal]))

%%
p=sym('p',[1,4]);
t=sym('t');
pt=p*getT(3,t);
vt=diff(pt,t)
at=diff(vt,t)
jt=diff(at,t)

vel_cost=int(vt*vt,t,0,1)
accel_cost=int(at*at,t,0,1)

total_cost=vel_cost+accel_cost;

H = double(hessian(total_cost)/2);

eig(H)

% p = [p(1)];
% q = polyder(p)