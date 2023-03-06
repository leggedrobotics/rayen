import numpy as np
import torch
import torch.nn as nn
import math

import matplotlib.pyplot as plt
import utils
from linear_constraint_walker import LinearConstraintWalker



use_example_3d=True

if(use_example_3d):
	A=np.array([ [1, 0, 0],
				 [0, 1, 0],
				 [0, 0, 1],
				 [-1, 0, 0],
				 [0, -1, 0],
				 [0, 0, -1]]);

	b=np.array([[1],
				[1],
				[1],
				[0],
				[0],
				[0]])


	Aeq=np.array([[1, 1, 1]]);
	beq=np.array([[1]]);

else:
	A=np.array([[-1,0],
				 [0, -1],
				 [0, 1],
				 [0.2425,    0.9701]]);

	b=np.array([[0],
				[0],
				[1],
				[1.2127]])

	Aeq=None;
	beq=None;


fig = plt.figure()
if(A.shape[1]==3):
	ax = fig.add_subplot(111, projection="3d")
	utils.plot3DPolytopeHRepresentation(A,b,[-1, 2, -1, 2, -1, 2], ax)
else:
	ax = fig.add_subplot(111) 

num_steps=2;
my_layer=LinearConstraintWalker(A, b, Aeq, beq, num_steps,use_max_ellipsoid=True)

numel_input_walker=my_layer.getNumelInputWalker()

##This puts everything in a batch and performs one call
all_angles=np.arange(0,2*math.pi, 0.01)
x_batched=torch.empty(len(all_angles), numel_input_walker, 1)
for i in range(x_batched.shape[0]): #for each element of the batch
	theta=all_angles[i]
	tmp=torch.Tensor(np.array([[math.cos(theta)],[math.sin(theta)],[3000]])); #Assumming my_layer.dim==2 here
	tmp= tmp.repeat(num_steps, 1)
	x_batched[i,:,:]=tmp


# mapper=nn.Sequential(nn.Linear(x_batched.shape[1], numel_input_walker))
mapper=nn.Sequential() #do nothing.
my_layer.setMapper(mapper)

result=my_layer(x_batched)

# my_layer.plotAllSteps(ax)

print(f"result={result}");
print(f"result.shape={result.shape}");
result=result.detach().numpy();


if(A.shape[1]==3):
	ax.scatter3D(result[:,0,0], result[:,1,0], result[:,2,0])

if(A.shape[1]==2):
	ax.scatter(result[:,0,0], result[:,1,0])
	utils.plot2DPolyhedron(A,b,ax)
	utils.plot2DEllipsoidB(my_layer.B[0,:,:].numpy(),my_layer.x0[0,:,:].numpy(),ax)
	ax.set_aspect('equal')

plt.show()
# # plot
# ax.scatter(result[:,0,0], result[:,1,0])


# plt.show()