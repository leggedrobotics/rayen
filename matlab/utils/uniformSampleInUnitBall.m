function result=uniformSampleInUnitBall(dim,num_points)
%Method 20 of http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

u = normrnd(0,1,dim,num_points);  % each column is an array of dim normally distributed random variables
u_normalized=u./vecnorm(u);
r = rand(1,num_points).^(1.0/dim); %each column is the radius of each of the points
result= r.*u_normalized;

end