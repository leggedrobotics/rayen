import torch
import torch.nn as nn
import utils
import numpy as np
import scipy
import cvxpy as cp
import math
from cvxpylayers.torch import CvxpyLayer


class LinearConstraintLayer(torch.nn.Module):
	def __init__(self, lc, method='walker'): #lc is the linear constraint
		super().__init__()

		assert cp.__version__=='1.2.3' #See this issue: https://github.com/cvxgrp/cvxpylayers/issues/143

		self.method=method

		if(self.method=='walker' or self.method=='barycentric' or self.method=='proj_train_test' or self.method=='proj_test'):
			A_p, b_p, y0, NA_E, B, z0=lc.process();

			self.Z_is_unconstrained= (not np.any(A_p))  and (not np.any(b_p)); #A_p is the zero matrix and b_p the zero vector


			self.dim=A_p.shape[1]
			A_p = torch.Tensor(A_p)
			b_p = torch.Tensor(b_p)
			y0=torch.Tensor(y0)
			NA_E=torch.Tensor(NA_E)
			B = torch.Tensor(B)
			z0 = torch.Tensor(z0)
			if(self.Z_is_unconstrained==False):
			 	D=torch.div(A_p,   (torch.sub(b_p,A_p@z0))@torch.ones(1,self.dim)    )
			else:
				D=torch.zeros_like(A_p)

			#See https://discuss.pytorch.org/t/model-cuda-does-not-convert-all-variables-to-cuda/114733/9
			# and https://discuss.pytorch.org/t/keeping-constant-value-in-module-on-correct-device/10129
			self.register_buffer("A_p", A_p)
			self.register_buffer("b_p", b_p)
			self.register_buffer("y0", y0)
			self.register_buffer("NA_E", NA_E)
			self.register_buffer("B", B)
			self.register_buffer("z0", z0)
			self.register_buffer("D", D)


			if(self.method=='proj_train_test' or self.method=='proj_test'):
				#Section 8.1.1 of https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
				self.x_projected = cp.Variable((self.dim,1))         #projected point
				self.x_to_be_projected = cp.Parameter((self.dim,1))  #original point
				constraints=[A_p@self.x_projected<=b_p]
				objective = cp.Minimize(cp.sum_squares(self.x_projected - self.x_to_be_projected))
				self.prob_projection = cp.Problem(objective, constraints)

				assert self.prob_projection.is_dpp()
				self.proj_layer = CvxpyLayer(self.prob_projection, parameters=[self.x_to_be_projected], variables=[self.x_projected])



			if(self.method=='barycentric'):
				print(f"A_p={A_p}")
				print(f"b_p={b_p}")
				print("Computing vertices and rays...")
				self.V,self.R = utils.H_to_V(A_p, b_p);
				self.V = torch.Tensor(self.V)
				self.R = torch.Tensor(self.R)
				self.num_vertices=self.V.shape[1];
				self.num_rays=self.R.shape[1];
				assert (self.num_vertices+self.num_rays)>0
				print(self.V)
				print(self.R)
				# exit()


		self.mapper=nn.Sequential();
		self.lc=lc;

	def getNumelOutputMapper(self):
		if(self.method=='walker'):
			return (self.dim+1)
		if(self.method=='unconstrained'):
			return self.lc.dimAmbSpace()
		if(self.method=='barycentric'):
			return self.num_vertices + self.num_rays
		if(self.method=='proj_train_test' or self.method=='proj_test'):
			return self.dim


	def setMapper(self, mapper):
		self.mapper=mapper
		utils.printInBoldRed(f"Setting the following mapper: {self.mapper}")

	def liftBack(self, z0_new):
		return self.NA_E@z0_new + self.y0

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
			
			v = z[:,  0:self.dim,0:1] #0:1 to keep the dimension. Other option is torch.unsqueeze(z[:,  0:self.dim,0],2) 
			beta= z[:, self.dim:(self.dim+1),0:1]#0:1 to keep the dimension. Other option istorch.unsqueeze(z[:, self.dim:(self.dim+1),0],2)
			
			u=torch.nn.functional.normalize(v, dim=1);


			## FIRST OPTION
			# bp_minus_Apz0=torch.sub(self.b_p,self.A_p@self.z0)
			# all_max_distances=torch.div(bp_minus_Apz0,self.A_p@u)
			# all_max_distances[all_max_distances<=0]=float("Inf")
			# #Note that we know that self.x0 is a strictly feasible point of the set
			# max_distance = torch.min(all_max_distances, dim=1, keepdim=True).values

			# #Here, the size of max_distance is [num_batches, 1, 1]

			# alpha=torch.where(torch.isfinite(max_distance), 
			# 				   max_distance*torch.sigmoid(beta),  #If it's bounded in that direction --> apply sigmoid function
			# 				   torch.abs(beta)) #If it's not bounded in that direction --> just use the beta



			## SECOND OPTION
			# if(self.Z_is_unconstrained==False):
			# 	my_lambda=torch.max(torch.div(self.A_p@u, bp_minus_Apz0), dim=1, keepdim=True).values
			# else:
			# 	my_lambda=torch.zeros((x.shape[0],1,1))

			# alpha=torch.where(my_lambda<=0, torch.abs(beta), torch.sigmoid(beta)/my_lambda)



			## THIRD OPTION
			kappa=torch.relu( torch.max(self.D@u, dim=1, keepdim=True).values  )
			alpha=1/(torch.exp(beta) + kappa)


			z0_new = self.z0 + alpha*u 
			
			#Now lift back to the original space
			y0_new =self.liftBack(z0_new)

			if(torch.isnan(y0_new).any()):
				print("at least one element is nan")
				raise Exception("exiting")

		if(self.method=='unconstrained'):
			y0_new=z

		if (self.method=='barycentric'):
			tmp1 = z[:,  0:self.num_vertices,0:1] #0:1 to keep the dimension. Other option is torch.unsqueeze(z[:,  0:self.dim,0],2) 
			tmp2 = z[:,  self.num_vertices:(self.num_vertices+self.num_rays),0:1] #0:1 to keep the dimension. Other option is torch.unsqueeze(z[:,  0:self.dim,0],2) 
			
			lambdas=nn.functional.softmax(tmp1, dim=1)
			mus=torch.abs(tmp2)

			z0_new=self.V@lambdas + self.R@mus

			print(z0_new.shape)
			#Now lift back to the original space
			y0_new =self.liftBack(z0_new)

		if(self.method=='proj_train_test'):
			z0_new, = self.proj_layer(z)
			#Now lift back to the original space
			y0_new =self.liftBack(z0_new)

		if(self.method=='proj_test'):
			if(self.training==False):
				z0_new, = self.proj_layer(z)
			else:
				z0_new = z

			y0_new =self.liftBack(z0_new)



		return y0_new
