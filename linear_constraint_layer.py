import torch
import torch.nn as nn
import utils
import numpy as np
import scipy
import cvxpy as cp
import math


class LinearConstraintLayer(torch.nn.Module):
	def __init__(self, lc, method='walker'): #lc is the linear constraint
		super().__init__()


		self.method=method

		if(self.method=='walker'):
			A_p, b_p, y0, NA_E, B, z0=lc.process();

			self.Z_is_unconstrained= (not np.any(A_p))  and (not np.any(b_p)); #A_p is the zero matrix and b_p the zero vector


			self.dim=A_p.shape[1]
			self.A_p = torch.Tensor(A_p)
			self.b_p = torch.Tensor(b_p)
			self.y0=torch.Tensor(y0)
			self.NA_E=torch.Tensor(NA_E)
			self.B = torch.Tensor(B)
			self.z0 = torch.Tensor(z0)



		self.mapper=nn.Sequential();
		self.lc=lc;

	def getNumelOutputMapper(self):
		if(self.method=='walker'):
			return (self.dim+1)
		if(self.method=='unconstrained'):
			return self.lc.dimAmbSpace()

	def setMapper(self, mapper):
		self.mapper=mapper
		utils.printInBoldRed(f"Setting the following mapper: {self.mapper}")

	def forward(self, x):

		# print("In forward pass")


		##################  MAPPER LAYER ####################
		# x has dimensions [num_batches, numel_input_mapper, 1]
		y = x.view(x.size(0), -1)
		# y has dimensions [num_batches, numel_input_mapper] This is needed to be able to pass it through the linear layer
		z = self.mapper(y)
		#Here z has dimensions [num_batches, numel_output_mapper]
		z = torch.unsqueeze(z,dim=2)
		#Here z has dimensions [num_batches, numel_output_mapper, 1]
		####################################################

		if(self.method=='walker'):

			self.A_p = self.A_p.to(x.device)
			self.b_p = self.b_p.to(x.device)
			self.B = self.B.to(x.device)
			self.z0 = self.z0.to(x.device)
			self.NA_E = self.NA_E.to(x.device)
			self.y0 = self.y0.to(x.device)
			
			v = z[:,  0:self.dim,0:1] #0:1 to keep the dimension. Other option is torch.unsqueeze(z[:,  0:self.dim,0],2) 
			beta= z[:, self.dim:(self.dim+1),0:1]#0:1 to keep the dimension. Other option istorch.unsqueeze(z[:, self.dim:(self.dim+1),0],2)
			
			u=torch.nn.functional.normalize(v, dim=1);

			bp_minus_Apz0=torch.sub(self.b_p,self.A_p@self.z0)

			print(bp_minus_Apz0)
			## FIRST OPTION
			# all_max_distances=torch.div(bp_minus_Apz0,self.A_p@u)
			# all_max_distances[all_max_distances<=0]=float("Inf")
			# #Note that we know that self.x0 is a strictly feasible point of the set
			# max_distance = torch.min(all_max_distances, dim=1, keepdim=True).values

			# #Here, the size of max_distance is [num_batches, 1, 1]

			# alpha=torch.where(torch.isfinite(max_distance), 
			# 				   max_distance*torch.sigmoid(beta),  #If it's bounded in that direction --> apply sigmoid function
			# 				   torch.abs(beta)) #If it's not bounded in that direction --> just use the beta

			## SECOND OPTION
			if(self.Z_is_unconstrained==False):
				my_lambda=torch.max(torch.div(self.A_p@u, bp_minus_Apz0), dim=1, keepdim=True).values
			else:
				my_lambda=torch.zeros((x.shape[0],1,1))

			alpha=torch.where(my_lambda<=0, torch.abs(beta), torch.sigmoid(beta)/my_lambda)


			z0_new = self.z0 + alpha*u 
			
			#Now lift back to the original space
			y0_new =self.NA_E@z0_new + self.y0

			if(torch.isnan(y0_new).any()):
				print("at least one element is nan")
				raise Exception("exiting")

		if(self.method=='unconstrained'):
			y0_new=z


		return y0_new
