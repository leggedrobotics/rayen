function plotConvexHullOfPoints(V,color)

x=V(1,:);
y=V(2,:);
z=V(3,:);

[k1,av1] = convhull(x,y,z);

trisurf(k1,x,y,z,'FaceColor',color)
k_all=k1(:);
k_all_unique=unique(k1);

for i=1:length(k_all_unique)
    plotSphere([x(k_all_unique(i)) y(k_all_unique(i)) z(k_all_unique(i))], 0.03, color)
end

end