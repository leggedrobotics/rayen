function [allA, allb, allV, p0, t0,tf,deg_pos, num_seg, num_of_seg_per_region, use_quadratic]=getCorridorAndParamsSpline(dimension)

    if(dimension==2)
        rng('default'); rng(6); %So that random is repeatable
%         P=3*[0 5.5 7.5 10.5 12.5;
%              0   4   0   0    4 ];
         P=3*[0 5.5 7.5  12.5;
             0   4   0     4 ];
        radius=4.0;
        num_of_seg_per_region=2; 
        samples_per_step=5;
        use_quadratic=false;    
        tf=35.0;
        deg_pos=2; %Note that I'm including the jerk cost. If I use deg_pos=2, then the jerk cost will always be zero
    else
        rng('default'); rng(2); %So that random is repeatable
        P=3*[0 1 2 3 4 3 0;
           0 1 1 2 4 4 4;
           0 1 1 1 4 1 0];
        radius=4*1.3;
        num_of_seg_per_region=2; 
        samples_per_step=3;
        use_quadratic=true;
        tf=15.0;
        deg_pos=3; %Note that I'm including the jerk cost. If I use deg_pos=2, then the jerk cost will always be zero
        p0=[0.5;0];
    end

    t0=0.0;
    
    allA={};
    allb={};
    allV={};
    
    steps=2;
    
    for i=1:(size(P,2)-1)
        if(dimension==3)
            [A, b, V]=getABVerticesgivenP1P2(P(:,i),P(:,i+1), 1.0, 1.0, 2);
        else
            [A, b, V]=getAbVerticesPolyhedronAroundP1P2(P(:,i),P(:,i+1), steps, samples_per_step, radius);
        end
    %     
        allA{end+1}=A;
        allb{end+1}=b;
        allV{end+1}=V;
    end

    num_of_regions=size(allA,2);
    num_seg =num_of_seg_per_region*num_of_regions; %First region has only 1 segment
    
     %  0.8*P(:,1) + 0.2*P(:,2);
%     pf=mean(allV{end},2); %0.2*P(:,end-1) + 0.8*P(:,end);

    if(dimension==2)
        p0=[5.0;1];
    else
        p0=mean(allV{1},2);
    end

    figure;
    hold on;
    alpha=0.2;
    
    for i=1:size(allA,2)
        plotregion(-allA{i},- allb{i}, [], [],'g',alpha)
    end
    
    % plotPolyhedron(P,'r')
    camlight
    lighting phong
    
    if(dimension==2)
        delta=4.0;
        xlim([min(P(1,:))-delta,max(P(1,:))+delta]);
        ylim([min(P(2,:))-delta,max(P(2,:))+delta]);
    else
        delta=4.0;
        xlim([min(P(1,:))-delta,max(P(1,:))+delta]);
        ylim([min(P(2,:))-delta,max(P(2,:))+delta]);
        zlim([min(P(3,:))-delta,max(P(3,:))+delta]);
        view(-71,40)
        xlabel('x')
        ylabel('y')
        zlabel('z')
    end
    
    if(dimension==2)
        scatter(p0(1),p0(2),'filled','b')
%         scatter(pf(1),pf(2),'filled','r')
    end
    
    if(dimension==3)
        zlim([min(P(3,:))-delta,max(P(3,:))+delta]);
        plotSphere(p0,0.2,'b')
%         plotSphere(pf,0.2,'r')
    end

end

%This gives the vertices of a polyhedron around the line p1-->p2
function [A, b, V]=getAbVerticesPolyhedronAroundP1P2(p1,p2, steps, samples_per_step, radius)


    dimension=size(p1,1);

%     samples_per_step=5;
%     radius=0.5;
    all_points=[];

    for alpha=linspace(0,1,steps)
        point=alpha*p1 + (1-alpha)*p2;
        results=point + radius*uniformSampleInUnitBall(dimension,samples_per_step);
        all_points=[all_points results];
    end

    [k1,av1] = convhull(all_points');
    k_all_unique=unique(k1);
    
    V=all_points(:,k_all_unique); %V contains the vertices

    [A,b,Aeq,beq]=vert2lcon(V');

    assert(numel(Aeq)==0)
    assert(numel(beq)==0)

%     for tmp=k_all_unique'
%         tmp
%         P(:,tmp)
%         plotSphere(P(:,tmp), 0.03, 'r');
%     end


end