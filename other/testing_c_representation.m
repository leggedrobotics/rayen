

A=rand(5,2);
B=rand(2,2);
c=squared_norm_each_row(A*B)


function result=squared_norm_each_row(D)
    result=(D.^2)*ones(size(D,2),1)
end