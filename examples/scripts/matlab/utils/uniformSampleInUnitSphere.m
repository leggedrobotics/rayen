function result=uniformSampleInUnitSphere(dim,num_points)
%Method 19 of http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

u = normrnd(0,1,dim,num_points);  % each column is an array of dim normally distributed random variables
u_normalized=u./vecnorm(u);
result= u_normalized;

end