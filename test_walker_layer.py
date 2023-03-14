import numpy as np
import torch
import torch.nn as nn
import math

import matplotlib.pyplot as plt
import utils
from linear_constraint_walker import LinearConstraintWalker
from examples_sets import getExample

lc=getExample(2)

fig = plt.figure()
if(lc.dimAmbSpace()==3):
	ax = fig.add_subplot(111, projection="3d")
	if(lc.hasIneqConstraints()):
		utils.plot3DPolytopeHRepresentation(lc.Aineq,lc.bineq,[-1, 2, -1, 2, -1, 2], ax)
else:
	ax = fig.add_subplot(111) 

num_steps=4; #Only used in the ellipsoid_walker method
my_layer=LinearConstraintWalker(lc)

numel_input_walker=my_layer.getNumelInputWalker()

##This samples different angles
# all_angles = np.arange(0,2*math.pi, 0.01)
# x_batched=torch.empty(len(all_angles), numel_input_walker, 1)

# for i in range(x_batched.shape[0]): #for each element of the batch
# 	theta=all_angles[i]
# 	if(my_layer.dim==2):
# 		tmp=torch.Tensor(np.array([[math.cos(theta)],[math.sin(theta)],[3000]])); #Assumming my_layer.dim==2 here
# 	else:
# 		raise("Not implemented yet")
# 	tmp=torch.unsqueeze(tmp, dim=0)
# 	print(f"tmp.shape={tmp.shape}")
# 	x_batched[i,:,:]=tmp


num_directions=500; #for each direction you have several samples
x_batched=torch.empty(0, numel_input_walker, 1)
for i in range(num_directions): #for each direction
	direction=utils.uniformSampleInUnitSphere(my_layer.dim)
	for scalar in list(np.linspace(-8.0, 8.0, num=100)):
		scalar_np=np.array([[scalar]])
		direction_and_scalar=np.concatenate((direction,scalar_np), axis=0);
		tmp=torch.Tensor(direction_and_scalar)
		tmp=torch.unsqueeze(tmp, dim=0)
		# print(f"direction_and_scalar={direction_and_scalar}")
		x_batched=torch.cat((x_batched, tmp), axis=0)


# print(f"x_batched={x_batched}")
# exit()
# mapper=nn.Sequential(nn.Linear(x_batched.shape[1], numel_input_walker))
mapper=nn.Sequential() #do nothing.
my_layer.setMapper(mapper)

result=my_layer(x_batched)

# my_layer.plotAllSteps(ax)

# print(f"result={result}");
# print(f"result.shape={result.shape}");
result=result.detach().numpy();


if(lc.Aineq.shape[1]==3):
	ax.scatter3D(result[:,0,0], result[:,1,0], result[:,2,0])

if(lc.Aineq.shape[1]==2):
	ax.scatter(result[:,0,0], result[:,1,0])
	utils.plot2DPolyhedron(lc.Aineq,lc.bineq,ax)
	utils.plot2DEllipsoidB(my_layer.B.numpy(),my_layer.x0.numpy(),ax)
	ax.set_aspect('equal')

plt.show()
# # plot
# ax.scatter(result[:,0,0], result[:,1,0])


# plt.show()