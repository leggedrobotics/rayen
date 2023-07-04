% --------------------------------------------------------------------------
% Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
% See LICENSE file for the license information
% -------------------------------------------------------------------------- 

function V_filtered=keepOnlyVerticesConvexHull(V)

    [k1,av1] = convhull(V');
    k_all_unique=unique(k1);
    V_filtered=V(:,k_all_unique); %V contains the vertices

end