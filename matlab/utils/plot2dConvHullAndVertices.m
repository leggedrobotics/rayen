function plot2dConvHullAndVertices(V1)
[k,av] = convhull(V1');
plot(V1(1,:),V1(2,:),'*')
hold on
plot(V1(1,k),V1(2,k))
end