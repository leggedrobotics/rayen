close all; clc;clear;
doSetup();
 set(0,'DefaultFigureWindowStyle','normal') %'normal' 'docked'

%Example of a plane and a plyhedron
radius=1.0;
% V=radius*uniformSampleInUnitBall(3,10);

V=[ 0.3345    0.1760   -0.4391    0.3967    0.0207   -0.7195    0.3139   -0.4449   -0.3855   -0.2624
    0.4675    0.7414   -0.0131   -0.2303    0.6983    0.0610   -0.0889   -0.1703    0.7373    0.7028
    0.1872   -0.5452   -0.5910   -0.0152    0.2959   -0.0226   -0.6567   -0.2332   -0.4466    0.2622];

V=keepOnlyVerticesConvexHull(V);
[A,b,Aeq,beq]=vert2lcon(V');
assert(numel(Aeq)==0); assert(numel(beq)==0)




% center=[0.5,0.5,0.5];
% side=[1.0 1.0 1.0];
% box =getAb_Box3D(center,side)
% A=box.A;
% b=box.b;
% [V,nr,nre]=lcon2vert(A,b,[],[])
% V=V'; %my convention

Aeq=[0.5 0.5 1];
beq=0.2;

plot3dConvHullAndVertices(V, 'g', 0.02)

syms x y z
tol_for_aliasing=0.0001;
plane_color=[0.98, 0.33, 0.06];
fimplicit3(Aeq*[x;y;z]-beq+tol_for_aliasing,[-0.5 0.4 -0.2 0.8 -1 1],'FaceColor',plane_color,'EdgeColor','none','FaceAlpha',0.3)

NAeq=null(Aeq);

%See slide 8-6 here: https://see.stanford.edu/materials/lsoeldsee263/08-min-norm.pdf
y1=pinv(Aeq)*beq
% plotSphere(x0,0.05, 'r');


%%%% Plot Ellipsoid (x-c)'*E*(x-c)<=1
% E=[3 0 0;
%    0 3 0;
%    0 0 5];
% c=[-0.5;0.8;0];
E=[5 0 0;
   0 5 0;
   0 0 7];
c=[-0.45;0.5;0];
syms x y z 
tmp=[x y z]';
P=2*E;
q=(-2*E*c)
r=c'*E*c-1

f=0.5*tmp'*P*tmp + q'*tmp + r ; 

ellipsoid_color=[0.94, 0.85, 1];
fimplicit3(  f   ,'EdgeColor','none','FaceAlpha',0.2 ,'MeshDensity',40 ,'LineWidth',0.0001,'FaceColor',ellipsoid_color)
%%%%



Ap=A*NAeq;
bp=b-A*y1;
[Vp,nr,nre]=lcon2vert(Ap,bp,[],[]);
Vp=Vp'; %my convention
figure;hold on;

% IF YOU WANNA VISUALIZE THE CONSTRAINTS:
% for i=1:size(Ap,1)
%     fimplicit(Ap(i,:)*[x;y]-bp(i),[-1 1 -1 1])
% end

plot2dConvHullAndVertices(Vp,'r','r');

%%%%%%%%%%%%

tmp=NAeq*[x y]' + y1;
f=0.5*tmp'*P*tmp + q'*tmp + r 
fimplicit(f,'Color','b','LineWidth',2.0)

%%%%%%%%%%%%%%%%%%%%%%

Vp_lifted=[];
for i=1:size(Vp,2)
    Vp_lifted=[Vp_lifted NAeq*Vp(:,i)+y1];
end
axis off
figure(2)
export_fig intersection.png -m2.5

figure(1)
shp = alphaShape(Vp_lifted(1,:)',Vp_lifted(2,:)',Vp_lifted(3,:)');
% plot(shp,'EdgeColor','none','FaceColor','r')
axis off; axis equal;
view([63.6 20.59])
set(gcf,'Position',[394         288        1603         981])


export_fig polyhedron3d.png -m2.5

% plotAxesArrowsT(length, w_T_b)

% myaa
% print('PeaksSurface','-dpng','-r400')
% handle=gcf;
% name_figure='test.pdf'
% set(handle,'Units','inches');
% screenposition = get(handle,'Position');
% set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',[screenposition(3:4)]);
% print (name_figure, '-dpdf', '-painters')
% exportgraphics(handle,name_figure,'BackgroundColor','none','ContentType','vector')
% set(gcf,'renderer','Painters')
% system(['pdfcrop ',name_figure,'.pdf ',name_figure,'.pdf']);


% myaa
% exportAsPdf(gca,'polyhedron_3d')

% lighting phong
% myaa %removes aliasing

% export_fig('polyhedron_3d','-pdf')
% 
% exportAsPdf(gcf,'polyhedron_3d')

% tmp=gca;
%  if (size(findobj(tmp.Children,'Type','Light'))<1) %If still no light in the subplot
%      camlight %create light
%  end
%  lighting phong

%  myaa
% fill3(Vp_lifted(1,:),Vp_lifted(2,:),Vp_lifted(3,:),'r')
% plot3(Vp_lifted(1,:), Vp_lifted(2,:), Vp_lifted(3,:))

%Note that the whole space is covered, but randomly sampling means that
%with a high probability, the sample will only fall in the middle of the
%polyhedron

% %Let's take a bunch of random 2d points in the triangle:
% all_q=[];
% for i=1:1000
%     lambda=rand(size(VVV,2),1); lambda=lambda/sum(lambda); %lambda has the barycentric coordinates
%     all_q=[all_q VVV*lambda];
% end
% 
% scatter(all_q(1,:), all_q(2,:));

%Let's lift them up to the original 3d space
% all_q_lifted=[];
% for i=1:size(all_q,2)
%     q=all_q(:,i);
%     q_lifted=NAeq*q+x0;
%     all_q_lifted=[all_q_lifted q_lifted];
% end
% 
% figure(1)
% scatter3(all_q_lifted(1,:), all_q_lifted(2,:), all_q_lifted(3,:));


