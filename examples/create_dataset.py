# Import libraries
import torch
from torch.utils.data import Dataset, DataLoader
from examples_sets import getExample
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.io
import utils
import time
import constraints

# create custom dataset class
class CustomDataset(Dataset):
	def __init__(self, all_x, all_y, all_Pobj, all_qobj, all_robj, all_times_s, all_costs):
		self.all_x = [torch.Tensor(item) for item in all_x] 
		self.all_y = [torch.Tensor(item) for item in all_y] 
		self.all_Pobj = [torch.Tensor(item) for item in all_Pobj] 
		self.all_qobj = [torch.Tensor(item) for item in all_qobj] 
		self.all_robj = [torch.Tensor(item) for item in all_robj] 
		self.all_times_s = [torch.Tensor(item) for item in all_times_s]
		self.all_costs = [torch.Tensor(item) for item in all_costs] 


	def __len__(self):
		return len(self.all_y)

	def __getitem__(self, idx):
		return self.all_x[idx], self.all_y[idx], self.all_Pobj[idx], self.all_qobj[idx], self.all_robj[idx], self.all_times_s[idx], self.all_costs[idx]

	def getNumelX(self):
		return self.all_x[0].shape[0] #Using the first element

	def getNumelY(self):
		return self.all_y[0].shape[0] #Using the first element

	# def plot(self, ax):
	# 	dim=self.all_x[0].shape[0]

	# 	# print(f"x[0]={x[0]}")

	# 	all_x_np=np.concatenate(self.all_x, axis=1 )
	# 	all_y_np=np.concatenate(self.all_y, axis=1 )

	# 	print(f"dim={dim}")
	# 	print(f"dim={dim}")


	# 	if(dim==3):
	# 		ax.scatter3D(all_x_np[0,:], all_x_np[1,:], all_x_np[2,:],color='red')
	# 		ax.scatter3D(all_y_np[0,:], all_y_np[1,:], all_y_np[2,:],color='blue')

	# 		print(f"all_x_np={all_x_np}")

	# 	if(dim==2):
	# 		ax.scatter(all_x_np[0,:], all_x_np[1,:],color='red')
	# 		ax.scatter(all_y_np[0,:], all_y_np[1,:],color='blue')

def createProjectionDataset(num_samples, cs, bbox_half_side): 

	all_x=[];
	all_y=[];
	all_Pobj=[];
	all_qobj=[];
	all_robj=[];
	all_times_s=[];
	all_costs=[];

	for i in range(num_samples):
		x=np.random.uniform(low=-bbox_half_side, high=bbox_half_side, size=(cs.k,1))
		all_x.append(x)
		start=time.time();
		y, cost = cs.project(x)
		all_times_s.append(time.time()-start)
		all_costs.append(cost)
		all_y.append(y)

		assert x.shape[1]==1

		# ||x-y||^2 = y'*y  -2x'*y + x'*x
		# Match with 0.5*y'*P_obj*y + q_obj'*y + r_obj 
		Pobj=2*np.eye(x.shape[0])
		qobj=-2*x
		robj=x.T@x

		assert qobj.shape[1]==1
		assert robj.shape[1]==1, f"robj.shape={robj.shape}"

		all_Pobj.append(Pobj)
		all_qobj.append(qobj)
		all_robj.append(robj)

	my_dataset = CustomDataset(all_x, all_y, all_Pobj, all_qobj, all_robj, all_times_s, all_costs)


	###plotting stuff
	# fig = plt.figure()
	# if(dim_ambient_space==3):
	# 	ax = fig.add_subplot(111, projection="3d")
	# else:
	# 	ax = fig.add_subplot(111)
	# my_dataset.plot(ax);
	# plt.show()
	#######
	# exit()

	return my_dataset

def getCorridorDatasetsAndConstraints(dimension):

	mat = scipy.io.loadmat('./matlab/corridor_dim'+str(dimension)+'.mat')

	all_x=list(mat["all_x"][0])
	all_y=list(mat["all_y"][0])
	all_Pobj=list(mat["all_Pobj"][0])
	all_qobj=list(mat["all_qobj"][0])
	all_robj=list(mat["all_robj"][0])
	all_times_s=list(mat["all_times_s"][0])
	all_costs=list(mat["all_costs"][0])

	all_x_out_dist=list(mat["all_x_out_dist"][0])
	all_y_out_dist=list(mat["all_y_out_dist"][0])
	all_Pobj_out_dist=list(mat["all_Pobj_out_dist"][0])
	all_qobj_out_dist=list(mat["all_qobj_out_dist"][0])
	all_robj_out_dist=list(mat["all_robj_out_dist"][0])
	all_times_s_out_dist=list(mat["all_times_s_out_dist"][0])
	all_costs_out_dist=list(mat["all_costs_out_dist"][0])

	# polyhedron=mat["polyhedron"]

	# A1=polyhedron['A1'][0,0];
	# b1=polyhedron['b1'][0,0];

	# polyhedron=mat["polyhedron"]

	# A1=polyhedron['A1'][0,0];
	# b1=polyhedron['b1'][0,0];

	A1=mat['A1'];
	b1=mat['b1'];

	A2=mat['A2'];
	b2=mat['b2'];

	if(len(mat["all_P"])>0):
		all_P=list(mat["all_P"][0])
		all_q=list(mat["all_q"][0])
		all_r=list(mat["all_r"][0])
	else:
		all_P=[]
		all_q=[]
		all_r=[]


	# print(f"Shape of A1={A1.shape}")
	# print(f"Shape of b1={b1.shape}")

	# # print(len(all_P))
	# # print(len(all_q))
	# # print(len(all_r))

	# print(all_P[0].shape)
	# print(all_q[0].shape)
	# print(all_r[0].shape)
	# exit()


	assert A1.ndim==2
	assert b1.ndim==2

	assert A2.ndim==2
	assert b2.ndim==2

	assert all_y[0].shape[1]==1
	assert all_x[0].shape[1]==1
	assert all_x_out_dist[0].shape[1]==1
	assert all_y_out_dist[0].shape[1]==1

	lc=constraints.LinearConstraint(A1=A1, b1=b1, A2=A2, b2=b2);

	qcs=[];
	for i in range(len(all_P)):
		qc=constraints.convexQuadraticConstraint(P=all_P[i], q=all_q[i], r=all_r[i]);
		qcs.append(qc)

	cs=constraints.convexConstraints(lc=lc, qcs=qcs, socs=[], sdpc=None)
	my_dataset = CustomDataset(all_x, all_y, all_Pobj, all_qobj, all_robj, all_times_s, all_costs)
	my_dataset_out_dist = CustomDataset(all_x_out_dist, all_y_out_dist, all_Pobj_out_dist, all_qobj_out_dist, all_robj_out_dist, all_times_s_out_dist, all_costs_out_dist)


	return my_dataset, my_dataset_out_dist, cs




