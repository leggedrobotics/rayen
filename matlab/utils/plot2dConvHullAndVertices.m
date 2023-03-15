function plot2dConvHullAndVertices(V1, color_vertices, color_sides)
[k,av] = convhull(V1');
scatter(V1(1,:),V1(2,:),color_vertices,'filled')
hold on
plot(V1(1,k),V1(2,k),color_sides,'LineWidth',2)
end