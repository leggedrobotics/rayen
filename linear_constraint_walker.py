import torch
import torch.nn as nn
import utils
import numpy as np
import scipy
import cvxpy as cp
import math


class LinearConstraintWalker(torch.nn.Module):
	def __init__(self, lc): #lc is the linear constraint
		super().__init__()

		A_projected, b_projected, p0, NA_eq_set, B, x0=lc.process();

		self.dim=A_projected.shape[1]

		self.A = torch.Tensor(A_projected)
		self.b = torch.Tensor(b_projected)
		self.p0=torch.Tensor(p0)
		self.NA_eq_set=torch.Tensor(NA_eq_set)
		self.B = torch.Tensor(B)
		self.x0 = torch.Tensor(x0)


		self.mapper=nn.Sequential();

	def getNumelInputWalker(self):
		return (self.dim+1)

	def setMapper(self, mapper):
		self.mapper=mapper
		utils.printInBoldRed(f"Setting the following mapper: {self.mapper}")

	def forward(self, x):

		# print("In forward pass")

		self.A = self.A.to(x.device)
		self.b = self.b.to(x.device)
		self.B = self.B.to(x.device)
		self.x0 = self.x0.to(x.device)
		self.NA_eq_set = self.NA_eq_set.to(x.device)
		self.p0 = self.p0.to(x.device)


		##################  MAPPER LAYER ####################
		# x has dimensions [num_batches, numel_input_mapper, 1]
		y = x.view(x.size(0), -1)
		# y has dimensions [num_batches, numel_input_mapper] This is needed to be able to pass it through the linear layer
		z = self.mapper(y)
		#Here z has dimensions [num_batches, numel_input_walker]
		z = torch.unsqueeze(z,dim=2)
		#Here z has dimensions [num_batches, numel_input_walker, 1]
		####################################################

		v = torch.unsqueeze(z[:,  0:self.dim,0],2) 
		scalar= torch.unsqueeze(z[:, self.dim:(self.dim+1),0],2)
		
		u=torch.nn.functional.normalize(v, dim=1);

		b_minus_Ax0=torch.sub(self.b,self.A@self.x0)
		all_max_distances=torch.div(b_minus_Ax0,self.A@u)
		all_max_distances[all_max_distances<=0]=float("Inf")
		#Note that we know that self.x0 is a strictly feasible point of the set
		max_distance = torch.min(all_max_distances, dim=1, keepdim=True).values

		#Here, the size of max_distance is [num_batches, 1, 1]

		alpha=torch.where(torch.isfinite(max_distance), 
						   max_distance*torch.sigmoid(scalar),  #If it's bounded in that direction --> apply sigmoid function
						   torch.abs(scalar)) #If it's not bounded in that direction --> just use the scalar

		x0_new = self.x0 + alpha*u 
		
		#Now lift back to the original space
		x0_new =self.NA_eq_set@x0_new + self.p0

		if(torch.isnan(x0_new).any()):
			print("at least one element is nan")
			raise("exiting")


		return x0_new
