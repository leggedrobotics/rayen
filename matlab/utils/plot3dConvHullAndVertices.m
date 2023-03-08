function plot3dConvHullAndVertices(V,radius_sphere)

    color='r';
    color_vertex=[.98 .45 .02];
    radius=radius_sphere;
    s={};
    for i=1:size(V,2)
        s{end+1}=plotSphere(V(:,i),radius, color_vertex); hold on;
        alpha(s{i},1.0)
    end
%     s2=plotSphere(v2,radius, color_vertex);
%     s3=plotSphere(v3,radius, color_vertex);
%     s4=plotSphere(v4,radius, color_vertex);
%     
%     alpha(s1,1.0)
%     alpha(s2,1.0)
%     alpha(s3,1.0)
%     alpha(s4,1.0)
    
%     shading faceted
%     hold on
     axis equal
     tmp=gca;
     if (size(findobj(tmp.Children,'Type','Light'))<1) %If still no light in the subplot
         camlight %create light
     end
     lighting phong

     vx=V(1,:);
     vy=V(2,:);
     vz=V(3,:);
    
    [k1,volume] = convhull(vx,vy,vz,'Simplify',true);

    s2=trisurf(k1,vx,vy,vz,'LineWidth',1,'FaceColor',color);
   
    alpha(s2,0.1)
    xlabel('x')
    ylabel('y')
    zlabel('z')

end