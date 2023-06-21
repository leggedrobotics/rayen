import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import numpy as np
import scipy
import os
import time

from examples_sets import getExample
import utils_examples

import fixpath #Following this example: https://github.com/tartley/colorama/blob/master/demos/demo01.py
from rayen import constraints, constraint_module, utils

methods=['RAYEN_old', 'RAYEN', 'Bar', 'UU', 'PP', 'UP', 'DC3']
index_examples_to_run=list(range(15))
###############

num_of_examples=len(index_examples_to_run)
rows=math.ceil(math.sqrt(num_of_examples))
cols=rows

for method in methods:
	utils.printInBoldRed(f"==================== METHOD: {method} ==========================")
	fig = plt.figure()
	fig.suptitle(method, fontsize=10)
	for i in range(num_of_examples):
		index_example=index_examples_to_run[i]
		utils.printInBoldGreen(f"==================== Example: {index_example} ")


		constraint=getExample(index_example)

		if(method=='Bar' and (constraint.has_quadratic_constraints or constraint.has_soc_constraints or constraint.has_lmi_constraints)):
			continue

		if(method=='DC3' and (constraint.has_soc_constraints or constraint.has_lmi_constraints)):
			continue

		# fig = plt.figure()
		if(constraint.k==3):
			ax = fig.add_subplot(rows,cols,i+1, projection="3d")
			if(constraint.has_linear_ineq_constraints):
				utils_examples.plot3DPolytopeHRepresentation(constraint.lc.A1,constraint.lc.b1,[-1, 2, -1, 2, -1, 2], ax)


		else:
			ax = fig.add_subplot(rows,cols,i+1) 

		if(method=='DC3'):
			args_DC3={}
			args_DC3['lr'] = 1e-4
			args_DC3['eps_converge'] = 1e-4
			args_DC3['momentum'] = 0.5
			args_DC3['max_steps_training'] = 10
			args_DC3['max_steps_testing'] = 50000 #float("inf")
		else:
			args_DC3 = None


		my_layer=constraint_module.ConstraintModule(constraint, method=method, create_map=False, args_DC3=args_DC3)

		numel_output_mapper=my_layer.getDimAfterMap()

		num_samples=500 #12000
		x_batched=torch.Tensor(num_samples, numel_output_mapper, 1).uniform_(-5.0, 5.0)

		# mapper=nn.Sequential(nn.Linear(x_batched.shape[1], numel_output_mapper))
		# mapper=nn.Sequential() #do nothing.
		# my_layer.setMapper(mapper)

		my_layer.eval() #This changes the self.training variable of the module

		time_start=time.time()
		result=my_layer(x_batched)
		total_time_per_sample= (time.time()-time_start)/num_samples

		result=result.detach().numpy();

		y0=my_layer.gety0();

		if(constraint.k==3):
			ax.scatter(y0[0,0], y0[1,0], y0[2,0],color='r',s=500)
			ax.scatter(result[:,0,0], result[:,1,0], result[:,2,0])


		if(constraint.k==2):
			ax.scatter(result[:,0,0], result[:,1,0])
			utils_examples.plot2DPolyhedron(constraint.lc.A1,constraint.lc.b1,ax)
			
			ax.scatter(y0[0,0], y0[1,0])
			# utils_examples.plot2DEllipsoidB(my_layer.B.numpy(),my_layer.z0.numpy(),ax)
			ax.set_aspect('equal')

			ax.set_xlim(-0.5,8)
			ax.set_ylim(-0.5,8)


		if method=='RAYEN':
			my_dict=constraint.getDataAsDict();
			my_dict["result"]=result
			my_dict["total_time_per_sample"]=total_time_per_sample
			directory='./examples_mat'
			if not os.path.exists(directory):
				os.makedirs(directory)
			scipy.io.savemat(directory+'/example_'+str(index_example)+'.mat', my_dict)

		utils.printInBoldBlue(f"Example {index_example}, total_time_per_sample={total_time_per_sample}")


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


# E=1.7*np.eye(constraint.k)
# ellipsoid=utils.Ellipsoid(E=E, c=np.zeros((constraint.k,1)))
# cqc_list=[ellipsoid.convertToQuadraticConstraint()]

# print(f"P={cqc_list[0].P}")
# print(f"q={cqc_list[0].q}")
# print(f"r={cqc_list[0].r}")


# if(method=='RAYEN'):
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

# if(method=='Bar' or method=='PP' or method=='UP'):
# 	x_batched=torch.empty(0, numel_output_mapper, 1)

# 	for i in range(1000):
# 		# sample_lambda = utils.runif_in_simplex(my_layer.num_vertices);
# 		# sample_mu = np.random.uniform(0.0,2.5,my_layer.num_rays);
# 		# sample=np.concatenate((sample_lambda, sample_mu));
# 		sample=np.random.uniform(-5,5,numel_output_mapper);
# 		sample=torch.Tensor(np.expand_dims(sample, axis=1))
# 		sample=torch.unsqueeze(sample, dim=0) 
# 		x_batched=torch.cat((x_batched, sample), axis=0) 
