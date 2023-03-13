# Import libraries
import torch
from torch.utils.data import Dataset, DataLoader
from examples_sets import getExample
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.io
import utils


# create custom dataset class
class CustomDataset(Dataset):
	def __init__(self, all_x, all_y):
		self.all_x = all_x
		self.all_y = all_y

	def __len__(self):
		return len(self.all_y)

	def __getitem__(self, idx):
		x = self.all_x[idx]
		y = self.all_y[idx]
		return x, y

	def getNumelX(self):
		return self.all_x[0].size #Using the first element

	def getNumelY(self):
		return self.all_y[0].size #Using the first element

	def plot(self, ax):
		dim=self.all_x[0].shape[0]

		# print(f"x[0]={x[0]}")

		all_x_np=np.concatenate(self.all_x, axis=1 )
		all_y_np=np.concatenate(self.all_y, axis=1 )

		print(f"dim={dim}")
		print(f"dim={dim}")


		if(dim==3):
			ax.scatter3D(all_x_np[0,:], all_x_np[1,:], all_x_np[2,:],color='red')
			ax.scatter3D(all_y_np[0,:], all_y_np[1,:], all_y_np[2,:],color='blue')

			print(f"all_x_np={all_x_np}")

		if(dim==2):
			ax.scatter(all_x_np[0,:], all_x_np[1,:],color='red')
			ax.scatter(all_y_np[0,:], all_y_np[1,:],color='blue')

def createProjectionDataset(num_samples, lc): #lc is the LinearConstraint

	all_x=[];
	all_y=[];

	
	for i in range(num_samples):
		x=np.random.uniform(low=-4.0, high=4.0, size=(lc.dimAmbSpace(),1))
		all_x.append(x)
		x_projected, _ = lc.project(x)
		all_y.append(x_projected)


	# define data set object
	my_dataset = CustomDataset(all_x, all_y)

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

def getCorridorDatasetAndLC():
	mat = scipy.io.loadmat('./matlab/corridor.mat')
	all_x=list(mat["all_x"][0])
	all_y=list(mat["all_y"][0])

	polyhedron=mat["polyhedron"]

	Aineq=polyhedron['Aineq'][0,0];
	bineq=polyhedron['bineq'][0,0];

	assert all_x[0].ndim==2
	assert Aineq.ndim==2
	assert bineq.ndim==2

	#This converts all the  to column vectors
	#Note that the order in the flatten() function needs to be consistent with the one from Casadi
	all_y_flattened=[np.expand_dims(y_i.flatten('F'), axis=1) for y_i in all_y] 
	all_x_flattened=all_x;

	assert all_y_flattened[0].shape[1]==1
	assert all_x_flattened[0].shape[1]==1

	lc=utils.LinearConstraint(Aineq, bineq, None, None)
	my_dataset = CustomDataset(all_x_flattened, all_y_flattened)

	return my_dataset, lc

	# print(polyhedron['Aineq'][0,0])

	# print(f"len(all_x)={len(all_x)}")
	# print(f"len(all_y)={len(all_y)}")
	# print(all_x[0].shape)
	# print(all_x[0])
	# print(all_y[0].shape)
	# print(all_y[0])


# print('\nFirst iteration of data set: ', next(iter(my_dataset)), '\n')
# print('Length of data set: ', len(my_dataset), '\n')
# print('Entire data set: ', list(DataLoader(my_dataset)), '\n')

# # create DataLoader object of DataSet object
# my_data_loader = DataLoader(my_dataset, batch_size=2, shuffle=True)

# # loop through each batch in the DataLoader object
# for (idx, batch) in enumerate(my_data_loader):
# 	print(f"Batch {idx} is {batch}")

# train_size = int(0.7 * len(my_dataset))
# validation_size = int(0.1 * len(my_dataset))
# test_size = len(my_dataset) - train_size - validation_size
# train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [train_size, validation_size, test_size])

# print(train_dataset.all_x)

# print(f"Length of train dataset={len(train_dataset)}")
# print(f"Length of test dataset={len(test_dataset)}")
# print(f"Length of validation dataset={len(validation_dataset)}")

