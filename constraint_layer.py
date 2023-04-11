import torch
import torch.nn as nn
import utils
import numpy as np
import scipy
import cvxpy as cp
import math
from cvxpylayers.torch import CvxpyLayer
import random

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



class CostComputer(nn.Module): #Using nn.Module to be able to use register_buffer (and hence to be able to have the to() method)
	def __init__(self, cs):
		super().__init__()

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

		self.has_quadratic_constraints=cs.has_quadratic_constraints

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
		all_inequalities=torch.cat((all_inequalities, self.A_p@z-self.b_p), dim=1);
		##### g(y)<=0
		if(self.has_quadratic_constraints):
			for i in range(self.all_P.shape[0]):
				P=self.all_P[i,:,:]
				q=self.all_q[i,:,:]
				r=self.all_r[i,:,:]
				all_inequalities=torch.cat((all_inequalities, utils.quadExpression(y=y,P=P,q=q,r=r)), dim=1)
		########################################################################

		return torch.sum(torch.square(torch.nn.functional.relu(all_inequalities)))


	def getSumObjCostAllSamples(self, y, Pobj, qobj, robj):
		tmp=utils.quadExpression(y=y,P=Pobj,q=qobj,r=robj)
		assert tmp.shape==(y.shape[0], 1, 1)

		return torch.sum(tmp)

	def getSumSupervisedCostAllSamples(self, y, y_predicted):
		return torch.sum(torch.square(y-y_predicted))

	def getSumLossAllSamples(self, params, y, y_predicted, Pobj, qobj, robj, isTesting=False):
		loss=params['use_supervised']*self.getSumSupervisedCostAllSamples(y, y_predicted) + \
			 (1-isTesting)*params['weight_soft_cost']*self.getSumSoftCostAllSamples(y_predicted) + \
			 (1-params['use_supervised'])*self.getSumObjCostAllSamples(y_predicted, Pobj, qobj, robj)

		return loss


class ConstraintLayer(torch.nn.Module):
	def __init__(self, cs, input_dim=None, method='walker_2', create_map=True, args_dc3=None):
		super().__init__()

		assert cp.__version__=='1.2.3' #See this issue: https://github.com/cvxgrp/cvxpylayers/issues/143

		self.method=method

		if(self.method=='barycentric' and cs.has_quadratic_constraints):
			raise Exception(f"Method {self.method} cannot be used with quadratic constraints")

		if(self.method=='dc3'):
			assert args_dc3 is not None
			self.args_dc3=args_dc3

		self.k=cs.k #Dimension of the ambient space
		self.n=cs.n #Dimension of the embedded space

		D=cs.A_p/((cs.b_p-cs.A_p@cs.z0)@np.ones((1,cs.n)))
			
		#See https://discuss.pytorch.org/t/model-cuda-does-not-convert-all-variables-to-cuda/114733/9
		# and https://discuss.pytorch.org/t/keeping-constant-value-in-module-on-correct-device/10129
		self.register_buffer("D", torch.tensor(D))
		self.register_buffer("all_P", torch.Tensor(np.array(cs.all_P)))
		self.register_buffer("all_q", torch.Tensor(np.array(cs.all_q)))
		self.register_buffer("all_r", torch.Tensor(np.array(cs.all_r)))
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
			V, R = utils.H_to_V(cs.A_p, cs.b_p);
			self.register_buffer("V", torch.tensor(V))
			self.register_buffer("R", torch.tensor(R))
			self.num_vertices=self.V.shape[1];
			self.num_rays=self.R.shape[1];
			assert (self.num_vertices+self.num_rays)>0
			print(f"vertices={self.V}")
			print(f"rays={self.R}")

		if(self.method=='dc3'):

			A2_dc3, b2_dc3=utils.removeRedundantEquationsFromEqualitySystem(cs.A_E, cs.b_E) 

			self.register_buffer("A2_dc3", torch.tensor(A2_dc3))
			self.register_buffer("b2_dc3", torch.tensor(b2_dc3))
			self.register_buffer("A1_dc3", torch.tensor(cs.A_I))
			self.register_buffer("b1_dc3", torch.tensor(cs.b_I))

			#Constraints are now 
			# A2_dc3 y = b2_dc3
			# A1_dc3 y <= b1_dc3

			det = 0
			i = 0
			self.neq_dc3 = self.A2_dc3.shape[0]
			while abs(det) < 0.0001 and i < 100:
				self.partial_vars = np.random.choice(self.k, self.k - self.neq_dc3, replace=False)
				self.other_vars = np.setdiff1d( np.arange(self.k), self.partial_vars)
				det = torch.det(self.A2_dc3[:, self.other_vars])
				i += 1
			if i == 100:
				raise Exception
			else:
				A2p = self.A2_dc3[:, self.partial_vars]
				A2oi = torch.inverse(self.A2_dc3[:, self.other_vars])			


			####################################################
			####################################################

			A1p=self.A1_dc3[:, self.partial_vars];
			A1o=self.A1_dc3[:, self.other_vars];

			A1_effective = A1p - A1o @ (A2oi @ A2p)
			b1_effective = self.b1_dc3 - A1o @ A2oi @ self.b2_dc3;

			all_P_effective=torch.Tensor(self.all_P.shape[0], len(self.partial_vars), len(self.partial_vars) )
			all_q_effective=torch.Tensor(self.all_q.shape[0], len(self.partial_vars), 1 )
			all_r_effective=torch.Tensor(self.all_q.shape[0], 1, 1 )

			self.register_buffer("A2oi", A2oi)
			self.register_buffer("A2p", A2p)
			self.register_buffer("A1_effective", A1_effective)
			self.register_buffer("b1_effective", b1_effective)

			####################
			for i in range(self.all_P.shape[0]): #for each of the quadratic constraints
				P=self.all_P[i,:,:]
				q=self.all_q[i,:,:]
				r=self.all_r[i,:,:]

				Po=P[np.ix_(self.other_vars,self.other_vars)].view(len(self.other_vars),len(self.other_vars))
				Pp=P[np.ix_(self.partial_vars,self.partial_vars)].view(len(self.partial_vars),len(self.partial_vars))
				Pop=P[np.ix_(self.other_vars,self.partial_vars)].view(len(self.other_vars),len(self.partial_vars))

				qo=q[self.other_vars,0:1]
				qp=q[self.partial_vars,0:1]

				b2=self.b2_dc3

				P_effective=2*(-A2p.T@A2oi.T@Pop + 0.5*A2p.T@A2oi.T@Po@A2oi@A2p + 0.5*Pp)
				q_effective=(b2.T@A2oi.T@Pop + qp.T - qo.T@A2oi@A2p - b2.T@A2oi.T@Po@A2oi@A2p).T
				r_effective=qo.T@A2oi@b2 + 0.5*b2.T@A2oi.T@Po@A2oi@b2 + r

				###### QUICK CHECK
				tmp=random.randint(1, 100) #number of elements in the batch
				yp=torch.rand(tmp, len(self.partial_vars), 1) 

				y = torch.zeros((tmp, self.k, 1))
				y[:, self.partial_vars, :] = yp
				y[:, self.other_vars, :] = self.obtainyoFromypDC3(yp)

				using_effective=utils.quadExpression(yp, P_effective, q_effective, r_effective)
				using_original=utils.quadExpression(y, P, q, r)

				# print(f"using_effective[0,0,0]={using_effective[0,0,0]}")
				# print(f"using_original[0,0,0]={using_original[0,0,0]}")
				# exit()

				assert torch.allclose(using_effective, using_original) 

				###################

				all_P_effective[i,:,:]=P_effective
				all_q_effective[i,:,:]=q_effective
				all_r_effective[i,:,:]=r_effective

			self.register_buffer("all_P_effective", all_P_effective)
			self.register_buffer("all_q_effective", all_q_effective)
			self.register_buffer("all_r_effective", all_r_effective)

			####################################################
			####################################################


		if(self.method=='walker_2'):
			self.forwardForMethod=self.forwardForWalker2
			self.dim_after_map=(self.n+1)
		elif(self.method=='walker_1'):
			self.forwardForMethod=self.forwardForWalker1
			self.dim_after_map=self.n
		elif(self.method=='unconstrained'):
			self.forwardForMethod=self.forwardForUnconstrained
			self.dim_after_map=self.k
		elif(self.method=='barycentric'):
			self.forwardForMethod=self.forwardForBarycentric
			self.dim_after_map=(self.num_vertices + self.num_rays)
		elif(self.method=='proj_train_test'):
			self.forwardForMethod=self.forwardForProjTrainTest
			self.dim_after_map=(self.n)
		elif(self.method=='proj_test'):
			self.forwardForMethod=self.forwardForProjTest
			self.dim_after_map=(self.n)
		elif(self.method=='dc3'):
			self.forwardForMethod=self.forwardForDc3
			self.dim_after_map=(self.k - self.neq_dc3)
		else:
			raise NotImplementedError

		if(create_map):
			assert input_dim is not None, "input_dim needs to be provided"
			self.mapper=nn.Linear(input_dim, self.dim_after_map);
		else:
			self.mapper=nn.Sequential(); #Mapper does nothing

	def obtainyoFromypDC3(self, yp):
		return self.A2oi @ (self.b2_dc3 - self.A2p @ yp)


	def forwardForDc3(self, q):
		
		#### Complete partial
		y = torch.zeros((q.shape[0], self.k, 1), device=q.device)
		y[:, self.partial_vars, :] = q
		y[:, self.other_vars, :] = self.obtainyoFromypDC3(q)

		#### Grad steps all

		y_new = y
		step_index = 0
		old_y_step = 0

		if(self.training):
			max_steps=self.args_dc3['max_steps_training'] #This is called corrTrainSteps in DC3 original code
		else:
			max_steps=self.args_dc3['max_steps_testing'] #float("inf") #This is called corrTestMaxSteps in DC3 original code

		while True:

			################################################
			################################################ COMPUTE y_step

			yp=y_new[:, self.partial_vars,:]
			ypT=torch.transpose(yp,1,2);

			grad = 2 * self.A1_effective.T @ torch.relu(self.A1_effective@yp - self.b1_effective)

			for i in range(self.all_P_effective.shape[0]): #for each of the quadratic constraints
				P_effective=self.all_P_effective[i,:,:]
				q_effective=self.all_q_effective[i,:,:]
				r_effective=self.all_r_effective[i,:,:]

				tmp1=(P_effective@yp + q_effective)
				tmp2=torch.relu(utils.quadExpression(yp, P_effective, q_effective, r_effective))

				grad += 2*tmp1@tmp2 #The 2 is because of the squared norm

			y_step = torch.zeros_like(y)
			y_step[:, self.partial_vars, :] = grad
			y_step[:, self.other_vars, :] = - self.A2oi @ self.A2p @ grad
			################################################
			################################################
			
			new_y_step = self.args_dc3['lr'] * y_step + self.args_dc3['momentum'] * old_y_step
			y_new = y_new - new_y_step
			old_y_step = new_y_step
			step_index += 1

			################################################
			################################################ COMPUTE current violation
			stacked=self.A1_dc3@y_new - self.b1_dc3
			for i in range(self.all_P.shape[0]): #for each of the quadratic constraints
				stacked=torch.cat((stacked,utils.quadExpression(y_new, self.all_P[i,:,:], self.all_q[i,:,:], self.all_r[i,:,:])), dim=1)
			violation=torch.max(torch.relu(stacked))
			################################################
			################################################

			# print(f"step_index={step_index}, Violation={violation}")

			converged_ineq = (violation < self.args_dc3['eps_converge'])
			max_iter_reached = (step_index >= max_steps)

			if(max_iter_reached):
				utils.printInBoldRed("Max iter reached")
				break

			if(converged_ineq):
				utils.printInBoldRed("Converged ineq reached")

				break


		return y_new



	def computeKappa(self,v_bar):

		kappa=torch.relu( torch.max(self.D@v_bar, dim=1, keepdim=True).values  )

		if(len(self.all_P)>0):
			nsib=v_bar.shape[0]; #number of samples in the batch
			lambda_quadratic=torch.empty((nsib,0,1), device=v_bar.device)
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

				tmp=(torch.min(lambda_quadratic, dim=1, keepdim=True).values)
				assert torch.all(lambda_quadratic >= 0)

				kappa = torch.maximum(kappa, 1.0/tmp)


		assert torch.all(kappa >= 0)

		return kappa

	def forwardForWalker2(self, q):
		v = q[:,  0:self.n,0:1]
		v_bar=torch.nn.functional.normalize(v, dim=1)
		kappa=self.computeKappa(v_bar)
		beta= q[:, self.n:(self.n+1),0:1]
		alpha=1/(torch.exp(beta) + kappa) 
		return self.getyFromz(self.z0 + alpha*v_bar)

	def forwardForWalker1(self, q):
		v = q[:,  0:self.n,0:1]
		v_bar=torch.nn.functional.normalize(v, dim=1)
		kappa=self.computeKappa(v_bar)
		norm_v=torch.linalg.vector_norm(v, dim=(1,2), keepdim=True)
		alpha=torch.minimum( 1/kappa , norm_v )
		return self.getyFromz(self.z0 + alpha*v_bar)

	def forwardForUnconstrained(self, q):
		return q

	def forwardForBarycentric(self, q):
		tmp1 = q[:,  0:self.num_vertices,0:1] #0:1 to keep the dimension. 
		tmp2 = q[:,  self.num_vertices:(self.num_vertices+self.num_rays),0:1] #0:1 to keep the dimension. 
		
		lambdas=nn.functional.softmax(tmp1, dim=1)
		mus=torch.abs(tmp2)

		return self.getyFromz(self.V@lambdas + self.R@mus)

	def project(self, q):
		#If you use ECOS, remember to set solver_args={'eps': 1e-6} (or smaller) for better solutions, see https://github.com/cvxpy/cvxpy/issues/880#issuecomment-557278620
		z, = self.proj_layer(q, solver_args={'solve_method':'ECOS'}) #Supported: ECOS (fast, accurate), SCS (slower, less accurate).   NOT supported: GUROBI
		return z

	def forwardForProjTrainTest(self, q):
		z=self.project(q)
		return self.getyFromz(z)


	def forwardForProjTest(self, q):
		if(self.training==False):
			z=self.project(q)
		else:
			z = q

		return self.getyFromz(z)

	def getDimAfterMap(self):
		return self.dim_after_map

	def gety0(self):
		return self.getyFromz(self.z0)

	def getyFromz(self, z):
		y=self.NA_E@z + self.y1
		return y

	def getzFromy(self, y):
		z=self.NA_E.T@(y - self.y1)
		return z

	def forward(self, x):

		##################  MAPPER LAYER ####################
		# x has dimensions [nsib, numel_input_mapper, 1]. nsib is the number of samples in the batch (i.e., x.shape[0]=x.shape[0])
		q = self.mapper(x.view(x.size(0), -1)) #After this, q has dimensions [nsib, numel_output_mapper]
		q = torch.unsqueeze(q,dim=2)  #After this, q has dimensions [nsib, numel_output_mapper, 1]
		####################################################

		y=self.forwardForMethod(q)

		assert (torch.isnan(y).any())==False

		return y