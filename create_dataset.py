# Import libraries
import torch
from torch.utils.data import Dataset, DataLoader
from examples_sets import getExample
import numpy as np
import cvxpy as cp
from linear_constraint_walker import checkAndGetDimAmbientSpace
import matplotlib.pyplot as plt


class Projector():
	def __init__(self, Aineq, bineq, Aeq, beq):

		has_ineq_constraints, has_eq_constraints, dim_ambient_space = checkAndGetDimAmbientSpace(Aineq, bineq, Aeq, beq);


		dim=Aineq.shape[1];
		self.y = cp.Variable((dim,1)) #y is the projected point
		self.x = cp.Parameter((dim,1))#x is the original point

		#Section 8.1.1 of https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf

		constraints=[]
		if(has_ineq_constraints):
			constraints.append(Aineq@self.y<=bineq)
		if(has_eq_constraints):
			constraints.append(Aeq@self.y==beq)

		objective = cp.Minimize(cp.sum_squares(self.y - self.x))
		self.prob = cp.Problem(objective, constraints)

	def project(self, x):
		print("Calling solve...")
		self.x.value=x;
		result = self.prob.solve(verbose=True);
		if(self.prob.status != 'optimal'):
			raise Exception("Value is not optimal")
		
		return self.y.value

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

def createProjectionDataset(num_samples, Aineq, bineq, Aeq, beq):

	all_x=[];
	all_y=[];

	has_ineq_constraints, has_eq_constraints, dim_ambient_space = checkAndGetDimAmbientSpace(Aineq, bineq, Aeq, beq);


	dim=Aineq.shape[1]
	projector=Projector(Aineq, bineq, Aeq, beq)
	for i in range(num_samples):
		x=np.random.uniform(low=-4.0, high=4.0, size=(dim,1))
		all_x.append(x)
		all_y.append(projector.project(x))


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

