close all; clc;clear;
set(0,'DefaultFigureWindowStyle','docked')%normal or docked
addpath(genpath('./../deep_panther/panther/matlab'))
addpath(genpath('./../deep_panther/submodules/minvo'))
addpath(genpath('./utils'))

center=[0.5,0.5,0.5];
side=[1.0 1.0 1.0];
box =getAb_Box3D(center,side)
A=box.A;
b=box.b;
[V,nr,nre]=lcon2vert(A,b,[],[])
V=V'; %my convention
Aeq=[1 1 1];
beq=1.0;



plot3dConvHullAndVertices(V, 0.02)

syms x y z
fimplicit3(Aeq*[x;y;z]-beq,[-1 1 -1 1 -1 1],'EdgeColor','none','FaceAlpha',.5)

NAeq=null(Aeq);

%See slide 8-6 here: https://see.stanford.edu/materials/lsoeldsee263/08-min-norm.pdf
x0=pinv(Aeq)*beq
plotSphere(x0,0.05, 'r');

AAA=A*NAeq;
bbb=b-A*x0;
[VVV,nr,nre]=lcon2vert(AAA,bbb,[],[])
VVV=VVV'; %my convention
figure;hold on;


for i=1:size(AAA,1)
    fimplicit(AAA(i,:)*[x;y]-bbb(i),[-1 1 -1 1])
end


plot2dConvHullAndVertices(VVV);

%Let's take a bunch of random 2d points in the triangle:
all_q=[];
for i=1:1000
    lambda=rand(3,1); lambda=lambda/sum(lambda); %lambda has the barycentric coordinates
    all_q=[all_q VVV*lambda];
end

scatter(all_q(1,:), all_q(2,:));

%Let's lift them up to the original 3d space
all_q_lifted=[];
for i=1:size(all_q,2)
    q=all_q(:,i);
    q_lifted=NAeq*q+x0;
    all_q_lifted=[all_q_lifted q_lifted];
end

figure(1)
scatter3(all_q_lifted(1,:), all_q_lifted(2,:), all_q_lifted(3,:));


