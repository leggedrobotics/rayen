%If radius=0, This gives a polyhedron with 6 faces around the line p1-->p2
%If radius>0, some extra points are taken around the vertices of the polyhedron
%Region is {x such that A1*x<=b1}
function [A, b, V]=getABVerticesgivenP1P2(p1,p2, hside, radius, num_samples_per_vertex)

    h=norm(p1-p2);
    
    
    b_p1_a=[hside -hside 0]';
    b_p1_b=[hside hside 0]';
    b_p1_c=[-hside hside 0]';
    b_p1_d=[-hside -hside 0]';
    
    b_p2_a=[hside -hside h]';
    b_p2_b=[hside hside h]';
    b_p2_c=[-hside hside h]';
    b_p2_d=[-hside -hside h]';
    
    yaw=0.0;
    
    zb=(p2-p1)/norm(p2-p1);
    xb=cross([-sin(yaw) cos(yaw) 0]',zb); 
    assert(norm(xb)>0)
    xb=xb/norm(xb);
    yb=cross(zb,xb);
    w_R_b=[xb yb zb];
    
    w_p1_a=w_R_b*b_p1_a+p1;
    w_p1_b=w_R_b*b_p1_b+p1;
    w_p1_c=w_R_b*b_p1_c+p1;
    w_p1_d=w_R_b*b_p1_d+p1;
    
    w_p2_a=w_R_b*b_p2_a+p1;
    w_p2_b=w_R_b*b_p2_b+p1;
    w_p2_c=w_R_b*b_p2_c+p1;
    w_p2_d=w_R_b*b_p2_d+p1;
    
    b_A=[1 0 0;
        0 1 0;
        0 0 1;
        -1 0 0;
        0 -1 0;
        0 0 -1];
    A=[];
    
    for i=1:size(b_A,1)
        A=[A;(w_R_b*b_A(i,:)')'];
    end
    
    b=[A(1,:)*w_p1_a;
       A(2,:)*w_p1_b;
       A(3,:)*w_p2_a;
       A(4,:)*w_p1_c;
       A(5,:)*w_p1_d;
       A(6,:)*w_p1_d;];
    
    [V,nr,nre]=lcon2vert(A,b,[],[]);
    V=V';
    

    all_samples=[];
    for i=1:size(V,2)
        vertex=V(:,i);
        all_samples_of_this_vertex=[];
        while size(all_samples_of_this_vertex,2)<num_samples_per_vertex  
            result = vertex + radius*uniformSampleInUnitSphere(3,1);
            is_inside_box=all((A*result-b)<=0);
            if(not(is_inside_box))
                all_samples_of_this_vertex=[all_samples_of_this_vertex result];
            end
        end
        all_samples=[all_samples all_samples_of_this_vertex];
    end
    
    [A,b,Aeq,beq]=vert2lcon(all_samples')
    V=all_samples;
end