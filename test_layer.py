import numpy as np
import torch
import torch.nn as nn
import math

import matplotlib.pyplot as plt
import utils
from linear_constraint_layer import LinearConstraintLayer
from examples_sets import getExample

lc=getExample(0)
method='walker' #'walker'

fig = plt.figure()
if(lc.dimAmbSpace()==3):
	ax = fig.add_subplot(111, projection="3d")
	if(lc.hasIneqConstraints()):
		utils.plot3DPolytopeHRepresentation(lc.Aineq,lc.bineq,[-1, 2, -1, 2, -1, 2], ax)
else:
	ax = fig.add_subplot(111) 

num_steps=4; #Only used in the ellipsoid_walker method
my_layer=LinearConstraintLayer(lc, method=method)

numel_output_mapper=my_layer.getNumelOutputMapper()



x_batched=torch.Tensor(1000, numel_output_mapper, 1).uniform_(-8, 8)

# if(method=='walker'):
# 	num_directions=500; #for each direction you have several samples
# 	x_batched=torch.empty(0, numel_output_mapper, 1)
# 	for i in range(num_directions): #for each direction
# 		direction=utils.uniformSampleInUnitSphere(my_layer.dim)
# 		for scalar in list(np.linspace(-8.0, 8.0, num=100)):
# 			scalar_np=np.array([[scalar]])
# 			direction_and_scalar=np.concatenate((direction,scalar_np), axis=0);
# 			tmp=torch.Tensor(direction_and_scalar)
# 			tmp=torch.unsqueeze(tmp, dim=0)
# 			# print(f"direction_and_scalar={direction_and_scalar}")
# 			x_batched=torch.cat((x_batched, tmp), axis=0)

# if(method=='barycentric' or method=='proj_train_test' or method=='proj_test'):
# 	x_batched=torch.empty(0, numel_output_mapper, 1)

# 	for i in range(1000):
# 		# sample_lambda = utils.runif_in_simplex(my_layer.num_vertices);
# 		# sample_mu = np.random.uniform(0.0,2.5,my_layer.num_rays);
# 		# sample=np.concatenate((sample_lambda, sample_mu));
# 		sample=np.random.uniform(-5,5,numel_output_mapper);
# 		sample=torch.Tensor(np.expand_dims(sample, axis=1))
# 		sample=torch.unsqueeze(sample, dim=0) 
# 		x_batched=torch.cat((x_batched, sample), axis=0) 


# mapper=nn.Sequential(nn.Linear(x_batched.shape[1], numel_output_mapper))
mapper=nn.Sequential() #do nothing.
my_layer.setMapper(mapper)

my_layer.eval() #This changes the self.training variable of the module
result=my_layer(x_batched)

result=result.detach().numpy();

if(lc.dimAmbSpace()==3):
	ax.scatter3D(result[:,0,0], result[:,1,0], result[:,2,0])

if(lc.dimAmbSpace()==2):
	ax.scatter(result[:,0,0], result[:,1,0])
	utils.plot2DPolyhedron(lc.Aineq,lc.bineq,ax)
	utils.plot2DEllipsoidB(my_layer.B.numpy(),my_layer.z0.numpy(),ax)
	ax.set_aspect('equal')

plt.show()

######OLD
##This samples different angles
# all_angles = np.arange(0,2*math.pi, 0.01)
# x_batched=torch.empty(len(all_angles), numel_output_mapper, 1)

# for i in range(x_batched.shape[0]): #for each element of the batch
# 	theta=all_angles[i]
# 	if(my_layer.dim==2):
# 		tmp=torch.Tensor(np.array([[math.cos(theta)],[math.sin(theta)],[3000]])); #Assumming my_layer.dim==2 here
# 	else:
# 		raise("Not implemented yet")
# 	tmp=torch.unsqueeze(tmp, dim=0)
# 	print(f"tmp.shape={tmp.shape}")
# 	x_batched[i,:,:]=tmp
