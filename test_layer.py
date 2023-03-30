import numpy as np
import torch
import torch.nn as nn
import math

import matplotlib.pyplot as plt
import utils
from linear_constraint_layer import LinearConstraintLayer
from examples_sets import getExample

import mpl_toolkits.mplot3d as a3

import matplotlib.colors as colors
import pylab as pl
import numpy as np

import scipy

# ax = a3.Axes3D(pl.figure())
# for i in range(10000):
#     vtx = np.random.rand(3,3)

# pl.show()


torch.set_default_dtype(torch.float64) ##Use float32 here??

index_example=0
lc=getExample(index_example)


method='walker' #'walker'

E=1.7*np.eye(lc.dimAmbSpace())
ellipsoid=utils.Ellipsoid(E=E, c=np.zeros((lc.dimAmbSpace(),1)))
cqc_list=[ellipsoid.convertToQuadraticConstraint()]

print(f"P={cqc_list[0].P}")
print(f"q={cqc_list[0].q}")
print(f"r={cqc_list[0].r}")

fig = plt.figure()
if(lc.dimAmbSpace()==3):
	ax = fig.add_subplot(111, projection="3d")
	if(lc.hasIneqConstraints()):
		utils.plot3DPolytopeHRepresentation(lc.Aineq,lc.bineq,[-1, 2, -1, 2, -1, 2], ax)

	# V=np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
	# tri = a3.art3d.Poly3DCollection([V])
	# tri.set_color(colors.rgb2hex(np.random.rand(3)))
	# tri.set_edgecolor('k')
	# ax.add_collection3d(tri)

else:
	ax = fig.add_subplot(111) 

num_steps=4; #Only used in the ellipsoid_walker method
my_layer=LinearConstraintLayer(lc, cqc_list=cqc_list, method=method)

numel_output_mapper=my_layer.getNumelOutputMapper()

x_batched=torch.Tensor(5000, numel_output_mapper, 1).uniform_(-8, 8)

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

y0=my_layer.gety0();

if(lc.dimAmbSpace()==3):
	ax.scatter(y0[0,0], y0[1,0], y0[2,0],color='r',s=500)
	ax.scatter(result[:,0,0], result[:,1,0], result[:,2,0])
	# ax.plot3D(y0[0,0], y0[1,0])
	print(f"y0={y0}")

	# for ellipsoid in ellipsoids:
	# 	if(ellipsoid.isESingular()==False):
	# 		utils.plotEllipsoid(ellipsoid.E, ellipsoid.c, ax)
	# 	else:
	# 		utils.printInBoldRed("E is singular, not plotting")


if(lc.dimAmbSpace()==2):
	ax.scatter(result[:,0,0], result[:,1,0])
	utils.plot2DPolyhedron(lc.Aineq,lc.bineq,ax)
	
	ax.scatter(y0[0,0], y0[1,0])
	# utils.plot2DEllipsoidB(my_layer.B.numpy(),my_layer.z0.numpy(),ax)
	ax.set_aspect('equal')


###################### SAVE TO MAT FILE

Aeq=lc.Aeq;
beq=lc.beq;
Aineq=lc.Aineq;
bineq=lc.bineq;

if(Aeq is None):
	Aeq=np.array([[]])
if(beq is None):
	beq=np.array([[]])
if(Aineq is None):
	Aineq=np.array([[]])
if(bineq is None):
	bineq=np.array([[]])

all_P=[];
all_q=[];
all_r=[];
for c in cqc_list:
	all_P.append(c.P)
	all_q.append(c.q)
	all_r.append(c.r)


scipy.io.savemat('example_'+str(index_example)+'.mat', dict(Aeq=Aeq, beq=beq, Aineq=Aineq, bineq=bineq, all_P=all_P, all_q=all_q, all_r=all_r, result=result))

################################################3


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
