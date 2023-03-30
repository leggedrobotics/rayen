import torch
import torch.nn as nn
import utils
import numpy as np
import scipy
import cvxpy as cp
import math
from cvxpylayers.torch import CvxpyLayer


class ConstraintLayer(torch.nn.Module):
	def __init__(self, cs, method='walker'):
		super().__init__()

		assert cp.__version__=='1.2.3' #See this issue: https://github.com/cvxgrp/cvxpylayers/issues/143

		self.method=method

		if(self.method=='barycentric' and cs.has_quadratic_constraints):
			utils.printInBoldRed(f"Method {self.method} cannot be used with quadratic constraints")
			exit();

		if(self.method=='walker' or self.method=='barycentric' or self.method=='proj_train_test' or self.method=='proj_test'):

			self.k=cs.k #Dimension of the embeded space

			if(cs.has_linear_constraints):
				if(cs.Z_is_unconstrained==False):
					D=cs.A_p/((cs.b_p-cs.A_p@cs.z0)@np.ones((1,cs.k))) 
				else:
					D=np.zeros_like(cs.A_p)
				
				self.register_buffer("D", torch.tensor(D))


			if(cs.has_quadratic_constraints):
				self.register_buffer("all_P", torch.Tensor(np.array(cs.all_P)))
				self.register_buffer("all_q", torch.Tensor(np.array(cs.all_q)))
				self.register_buffer("all_r", torch.Tensor(np.array(cs.all_r)))
		
			#See https://discuss.pytorch.org/t/model-cuda-does-not-convert-all-variables-to-cuda/114733/9
			# and https://discuss.pytorch.org/t/keeping-constant-value-in-module-on-correct-device/10129
			self.register_buffer("A_p", torch.tensor(cs.A_p))
			self.register_buffer("b_p", torch.tensor(cs.b_p))
			self.register_buffer("y1", torch.tensor(cs.y1))
			self.register_buffer("NA_E", torch.tensor(cs.NA_E))
			self.register_buffer("z0", torch.tensor(cs.z0))


			if(self.method=='proj_train_test' or self.method=='proj_test'):
				#Section 8.1.1 of https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
				self.z_projected = cp.Variable((cs.k,1))         #projected point
				self.z_to_be_projected = cp.Parameter((cs.k,1))  #original point
				constraints= cs.getConstraintsInSubspaceCvxpy(self.z_projected)
				objective = cp.Minimize(cp.sum_squares(self.z_projected - self.z_to_be_projected))
				self.prob_projection = cp.Problem(objective, constraints)

				assert self.prob_projection.is_dpp()
				self.proj_layer = CvxpyLayer(self.prob_projection, parameters=[self.z_to_be_projected], variables=[self.z_projected])



			if(self.method=='barycentric'):
				print(f"A_p={cs.A_p}")
				print(f"b_p={cs.b_p}")
				print("Computing vertices and rays...")
				self.V,self.R = utils.H_to_V(cs.A_p, cs.b_p);
				self.V = torch.Tensor(self.V)
				self.R = torch.Tensor(self.R)
				self.num_vertices=self.V.shape[1];
				self.num_rays=self.R.shape[1];
				assert (self.num_vertices+self.num_rays)>0
				print(f"vertices={self.V}")
				print(f"rays={self.R}")
				# exit()




		self.mapper=nn.Sequential();
		self.cs=cs;

	def getNumelOutputMapper(self):
		if(self.method=='walker'):
			return (self.cs.k+1)
		if(self.method=='unconstrained'):
			return self.lc.dimAmbSpace()
		if(self.method=='barycentric'):
			return self.num_vertices + self.num_rays
		if(self.method=='proj_train_test' or self.method=='proj_test'):
			return self.cs.k


	def setMapper(self, mapper):
		self.mapper=mapper
		utils.printInBoldRed(f"Setting the following mapper: {self.mapper}")

	def liftBack(self, z0_new):
		return self.NA_E@z0_new + self.y1

	def gety0(self):
		return self.liftBack(self.z0)

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
			
			v = z[:,  0:self.cs.k,0:1] #0:1 to keep the dimension. Other option is torch.unsqueeze(z[:,  0:self.cs.k,0],2) 
			beta= z[:, self.cs.k:(self.cs.k+1),0:1]#0:1 to keep the dimension. Other option istorch.unsqueeze(z[:, self.cs.k:(self.cs.k+1),0],2)
			
			u=torch.nn.functional.normalize(v, dim=1);


			## THIRD OPTION
			if(self.cs.has_linear_constraints):
				kappa_linear=torch.relu( torch.max(self.D@u, dim=1, keepdim=True).values  )

				assert torch.all(kappa_linear >= 0)


			if(self.cs.has_quadratic_constraints):
				for i in range(self.all_P.shape[0]): #for each of the quadratic constraints
					P=self.all_P[i,:,:]
					q=self.all_q[i,:,:]
					r=self.all_r[i,:,:]

					rho = self.NA_E@u
					w = self.NA_E@self.z0 + self.y1 #Do this in the constructor of this class

					rhoT=torch.transpose(rho,dim0=1, dim1=2)

					a=0.5*rhoT@P@rho;
					b=(w.T@P@rho + q.T@rho);
					c=(0.5*w.T@P@w + q.T@w +r)

					aprime=rhoT@rho;
					bprime=2*w.T@rho;
					cprime=w.T@w - 1;

					discriminant = torch.square(b) - 4*(a)*(c)

					assert torch.all(discriminant >= 0) 
					lamb_positive=torch.div(  -(b)  + torch.sqrt(discriminant) , 2*a)
					assert torch.all(lamb_positive >= 0) #If not, then either the feasible set is infeasible (note that z0 is inside the feasible set)
					
					kappa_quadratic=1/lamb_positive;

					assert torch.all(kappa_quadratic >= 0)


			################################# Obtain kappa
			if(self.cs.has_linear_constraints and self.cs.has_quadratic_constraints):
				kappa=torch.maximum(kappa_linear, kappa_quadratic)
			elif(self.cs.has_linear_constraints):
				kappa=kappa_linear
			elif(self.cs.has_quadratic_constraints):
				kappa=kappa_quadratic
			else:
				assert False, "There are no constraints"
		
			assert torch.all(kappa >= 0)
			#################################


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
			tmp1 = z[:,  0:self.num_vertices,0:1] #0:1 to keep the dimension. Other option is torch.unsqueeze(z[:,  0:self.cs.k,0],2) 
			tmp2 = z[:,  self.num_vertices:(self.num_vertices+self.num_rays),0:1] #0:1 to keep the dimension. Other option is torch.unsqueeze(z[:,  0:self.cs.k,0],2) 
			
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


			#########################################################3
			###For the convex quadratic constraints constraints
			# if(self.has_ellipsoid_constraints):
			# 	for i in range(self.all_E.shape[0]): #for each of the ellipsoids
			# 		E=self.all_E[i,:,:]
			# 		q=self.all_q[i,:,:]
			# 		r=self.NA_E@u;
			# 		rT=torch.transpose(r,dim0=1, dim1=2)
			# 		discriminant = torch.square(2*rT@E@q) - 4*(rT@E@r)*(q.T@E@q-1)
			# 		assert torch.all(discriminant >= 0) 
			# 		lamb_positive=torch.div(  -2*rT@E@q  + torch.sqrt(discriminant) , 2*rT@E@r)
			# 		assert torch.all(lamb_positive >= 0) #If not, then either the feasible set is infeasible, or z0 was taken outside the feasible set
			# 		kappa=torch.maximum(kappa, 1/lamb_positive)


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