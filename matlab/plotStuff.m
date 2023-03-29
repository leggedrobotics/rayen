clear;
close all;
clc;

load('./../example_0.mat')
Aeq=double(Aeq);
beq=double(beq);
Aineq=double(Aineq);
bineq=double(bineq);

result=result';

syms  x y z 
tol_for_aliasing=0.0001;
figure; hold on;

V=lcon2vert(Aineq,bineq,Aeq,beq)
if(size(V,2)==3)
    V=[V 1.00001*V(:,end)] %Hack to prevent convhull from failing
end
plot3dConvHullAndVertices(V, 'g', 0.02)


for i=1:size(all_E,1)
    E=squeeze(all_E(i,:,:));
    c=squeeze(all_c(i,:,:));
    tmp=[x y z]';
    f=(tmp-c)'*E*(tmp-c)-1  + 1e-100*sum(tmp); %The second part is just a hack to make fimplicit3 work correctly when f depends only on two variables (imagine a cylinder aligned with the x axis) 
    fimplicit3(  f   ,'EdgeColor','none','FaceAlpha',.5 ,'MeshDensity',50 )
end

zlim([0,2])

scatter3(result(1,:),result(2,:),result(3,:))


%%
close all; figure;
Q=eye(3)
tmp=[x y z]';
c=rand(3,1); d=rand()
f=0.5*tmp'*Q*x + c'*tmp +d  + 1e-100*sum(tmp); %The second part is just a hack to make fimplicit3 work correctly when f depends only on two variables (imagine a cylinder aligned with the x axis) 
fimplicit3(  f   ,'EdgeColor','none','FaceAlpha',.5 ,'MeshDensity',50 )
