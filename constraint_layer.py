import torch
import torch.nn as nn
import utils
import numpy as np
import scipy
import cvxpy as cp
import math
from cvxpylayers.torch import CvxpyLayer

# class OptimizationLayer(torch.nn.Module):
# 	def __init__(self, cs):

# 		variable = cp.Parameter((cs.k,1))  
# 		Pobj = cp.Parameter((cs.k,cs.k))  
# 		qobj = cp.Parameter((cs.k,1))  
# 		robj = cp.Parameter((1,1))  

# 		#Problem is https://github.com/cvxgrp/cvxpylayers/issues/136#issuecomment-1410563781
# 		objective=cp.Minimize(0.5*cp.quad_form(variable, Pobj) + qobj.T@variable + robj)
# 		constraints=cs.getConstraintsCvxpy(variable);

# 		self.prob_projection = cp.Problem(objective, constraints)

# 		self.problem = CvxpyLayer(self.prob_projection, parameters=[Pobj, qobj, robj], variables=[variable])

# 	def forward(self, Pobj, qobj, robj):
# 		solution, = self.problem(Pobj, qobj, robj)
# 		return solution


class ConstraintLayer(torch.nn.Module):
	def __init__(self, cs, method='walker_2'):
		super().__init__()

		assert cp.__version__=='1.2.3' #See this issue: https://github.com/cvxgrp/cvxpylayers/issues/143

		self.method=method

		if(self.method=='barycentric' and cs.has_quadratic_constraints):
			utils.printInBoldRed(f"Method {self.method} cannot be used with quadratic constraints")
			exit();

		# if(self.method=='walker' or self.method=='barycentric' or self.method=='proj_train_test' or self.method=='proj_test'):

		self.n=cs.n #Dimension of the embeded space

		if(cs.has_linear_constraints):
			D=cs.A_p/((cs.b_p-cs.A_p@cs.z0)@np.ones((1,cs.n)))
			
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
			self.z_projected = cp.Variable((cs.n,1))         #projected point
			self.z_to_be_projected = cp.Parameter((cs.n,1))  #original point
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
		if(self.method=='walker_2'):
			return (self.cs.n+1)
		if(self.method=='walker_1'):
			return (self.cs.n)
		if(self.method=='unconstrained'):
			return self.cs.k
		if(self.method=='barycentric'):
			return self.num_vertices + self.num_rays
		if(self.method=='proj_train_test' or self.method=='proj_test'):
			return self.cs.n


	def setMapper(self, mapper):
		self.mapper=mapper
		utils.printInBoldRed(f"Setting the following mapper: {self.mapper}")

	def gety0(self):
		return self.getyFromz(self.z0)

	def getyFromz(self, z):
		y=self.NA_E@z + self.y1
		return y

	def getzFromy(self, y):
		z=self.NA_E.T@(y - self.y1)
		return z

	def getSumSoftCostAllSamples(self, y):
		z=self.getzFromy(y);


		#Note that in the projected subspace we don't have equalities

		################## STACK THE VALUES OF ALL THE INEQUALITIES (Ap*z-bp<=0 and g(y)<=0)
		all_inequalities=torch.empty((y.shape[0],0,1), device=y.device)
		##### Ap*z<=bp
		if(self.cs.has_linear_constraints):
			all_inequalities=torch.cat((all_inequalities, self.A_p@z-self.b_p), dim=1);
		##### g(y)<=0
		if(self.cs.has_quadratic_constraints):
			for i in range(self.all_P.shape[0]):
				P=self.all_P[i,:,:]
				q=self.all_q[i,:,:]
				r=self.all_r[i,:,:]
				all_inequalities=torch.cat((all_inequalities, utils.quadExpression(y=y,P=P,q=q,r=r)), dim=1)
		########################################################################


		# #########################################################
		# nsib=y.shape[0]; #number of samples in the batch
		# assert nsib==all_inequalities.shape[0]
		# tmp=all_inequalities.shape[1]

		# violation_squared=torch.square(torch.nn.functional.relu(all_inequalities));     #violation_squared is [nsib, tmp, 1]
		# sum_violation_squared=torch.sum(violation_squared, dim=1, keepdim=True)         #sum_violation_squared is [nsib, 1, 1]
		
		# assert violation_squared.shape==(nsib, tmp, 1), f"violation_squared.shape={violation_squared.shape}"
		# assert sum_violation_squared.shape==(nsib, 1, 1), f"sum_violation_squared.shape={sum_violation_squared.shape}"
		# ###########################################################


		# assert torch.all(sum_violation_squared>=0)

		# # return torch.mean(sum_violation_squared)
		return torch.sum(torch.square(torch.nn.functional.relu(all_inequalities)))


	def getSumObjCostAllSamples(self, y, Pobj, qobj, robj):
		tmp=utils.quadExpression(y=y,P=Pobj,q=qobj,r=robj)
		assert tmp.shape==(y.shape[0], 1, 1)
		assert torch.all(tmp>=0)
		# return torch.mean(tmp)
		return torch.sum(tmp)

	def getSumSupervisedCostAllSamples(self, y, y_predicted):
		# difference_squared=torch.square(y-y_predicted);
		# sum_difference_squared=torch.sum(difference_squared, dim=1, keepdim=True) 

		# assert sum_difference_squared.shape==(y.shape[0], 1, 1)
		# assert torch.all(sum_difference_squared>=0)

		# return torch.mean(sum_difference_squared)
		return torch.sum(torch.square(y-y_predicted))

	def getSumLossAllSamples(self, params, y, y_predicted, Pobj, qobj, robj, isTesting=False):
		loss=params['use_supervised']*self.getSumSupervisedCostAllSamples(y, y_predicted) + \
			 (1-isTesting)*params['weight_soft_cost']*self.getSumSoftCostAllSamples(y_predicted) + \
			 (1-params['use_supervised'])*self.getSumObjCostAllSamples(y_predicted, Pobj, qobj, robj)

		assert loss>=0

		return loss


	def forward(self, x):

		device=x.device
		# print("In forward pass")

		nsib=x.shape[0]; #number of samples in the batch

		##################  MAPPER LAYER ####################
		# x has dimensions [nsib, numel_input_mapper, 1]
		y = x.view(x.size(0), -1)
		# y has dimensions [nsib, numel_input_mapper] This is needed to be able to pass it through the linear layer
		z = self.mapper(y)
		#Here z has dimensions [nsib, numel_output_mapper]
		z = torch.unsqueeze(z,dim=2)
		#Here z has dimensions [nsib, numel_output_mapper, 1]
		####################################################

		if(self.method=='walker_2' or self.method=='walker_1'):
			
			v = z[:,  0:self.cs.n,0:1] #0:1 to keep the dimension. Other option is torch.unsqueeze(z[:,  0:self.cs.n,0],2) 
			
			v_bar=torch.nn.functional.normalize(v, dim=1);


			kappa_linear=torch.zeros((nsib,1,1), device=device)
			kappa_quadratic=torch.zeros((nsib,1,1), device=device)

			if(self.cs.has_linear_constraints):
				kappa_linear=torch.relu( torch.max(self.D@v_bar, dim=1, keepdim=True).values  )

				assert torch.all(kappa_linear >= 0)


			if(self.cs.has_quadratic_constraints):
				lambda_quadratic=torch.empty((nsib,0,1), device=device)
				for i in range(self.all_P.shape[0]): #for each of the quadratic constraints
					P=self.all_P[i,:,:]
					q=self.all_q[i,:,:]
					r=self.all_r[i,:,:]

					rho = self.NA_E@v_bar
					w = self.NA_E@self.z0 + self.y1 #Do this in the constructor of this class. This is actually y0

					rhoT=torch.transpose(rho,dim0=1, dim1=2)

					a=0.5*rhoT@P@rho;
					b=(w.T@P@rho + q.T@rho);
					c=(0.5*w.T@P@w + q.T@w +r)

					discriminant = torch.square(b) - 4*(a)*(c)

					assert torch.all(discriminant >= 0) 
					lamb_positive_i=torch.div(  -(b)  + torch.sqrt(discriminant) , 2*a)
					assert torch.all(lamb_positive_i >= 0) #If not, then either the feasible set is infeasible (note that z0 is inside the feasible set)
					
					lambda_quadratic = torch.cat((lambda_quadratic, lamb_positive_i), dim=1)


				# print(f"Shape={lambda_quadratic.shape}")
				tmp=(torch.min(lambda_quadratic, dim=1, keepdim=True).values)
				kappa_quadratic=1.0/tmp
				assert torch.all(lambda_quadratic >= 0)


			################################# Obtain kappa
			kappa = torch.maximum(kappa_linear, kappa_quadratic)
		
			assert torch.all(kappa >= 0)
			#################################


			if(self.method=='walker_2'):
				beta= z[:, self.cs.n:(self.cs.n+1),0:1]#0:1 to keep the dimension. Other option istorch.unsqueeze(z[:, self.cs.n:(self.cs.n+1),0],2)

				alpha=1/(torch.exp(beta) + kappa)
			else: #method is walker_1
				norm_v=torch.linalg.vector_norm(v, dim=(1,2), keepdim=True)
				alpha=torch.minimum( 1/kappa , norm_v )

			z0_new = self.z0 + alpha*v_bar 
			
			#Now lift back to the original space
			y0_new =self.getyFromz(z0_new)

			if(torch.isnan(y0_new).any()):
				print("at least one element is nan")
				raise Exception("exiting")

		if(self.method=='unconstrained'):
			y0_new=z

		if (self.method=='barycentric'):
			tmp1 = z[:,  0:self.num_vertices,0:1] #0:1 to keep the dimension. Other option is torch.unsqueeze(z[:,  0:self.cs.n,0],2) 
			tmp2 = z[:,  self.num_vertices:(self.num_vertices+self.num_rays),0:1] #0:1 to keep the dimension. Other option is torch.unsqueeze(z[:,  0:self.cs.n,0],2) 
			
			lambdas=nn.functional.softmax(tmp1, dim=1)
			mus=torch.abs(tmp2)

			z0_new=self.V@lambdas + self.R@mus

			print(z0_new.shape)
			#Now lift back to the original space
			y0_new =self.getyFromz(z0_new)

		if(self.method=='proj_train_test'):
			z0_new, = self.proj_layer(z)
			#Now lift back to the original space
			y0_new =self.getyFromz(z0_new)

		if(self.method=='proj_test'):
			if(self.training==False):
				z0_new, = self.proj_layer(z)
			else:
				z0_new = z

			y0_new =self.getyFromz(z0_new)



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

			# #Here, the size of max_distance is [nsib, 1, 1]

			# alpha=torch.where(torch.isfinite(max_distance), 
			# 				   max_distance*torch.sigmoid(beta),  #If it's bounded in that direction --> apply sigmoid function
			# 				   torch.abs(beta)) #If it's not bounded in that direction --> just use the beta


			## SECOND OPTION
			# if(self.Z_is_unconstrained==False):
			# 	my_lambda=torch.max(torch.div(self.A_p@u, bp_minus_Apz0), dim=1, keepdim=True).values
			# else:
			# 	my_lambda=torch.zeros((x.shape[0],1,1))

			# alpha=torch.where(my_lambda<=0, torch.abs(beta), torch.sigmoid(beta)/my_lambda)