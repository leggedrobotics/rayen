function result=getAb_Box3D(center,side)

    A1=[1 0 0;
        0 1 0;
        0 0 1;
        -1 0 0;
        0 -1 0;
        0 0 -1];
    
    of_x=center(1);
    of_y=center(2);
    of_z=center(3);
    
    b1=[side(1)/2.0+of_x;
        side(2)/2.0+of_y;
        side(3)/2.0+of_z;
        side(1)/2.0-of_x;
        side(2)/2.0-of_y;
        side(3)/2.0-of_z];
    
    result.A=A1;
    result.b=b1;

end