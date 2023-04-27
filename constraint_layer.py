import torch
import torch.nn as nn
import utils
import numpy as np
import scipy
import cvxpy as cp
import math
from cvxpylayers.torch import CvxpyLayer
import random
import copy

class ConstraintLayer(torch.nn.Module):
	def __init__(self, cs, input_dim=None, method='walker_2', create_map=True, args_DC3=None):
		super().__init__()

		assert cp.__version__=='1.2.3' #See this issue: https://github.com/cvxgrp/cvxpylayers/issues/143

		self.method=method

		if(self.method=='Bar' and cs.has_quadratic_constraints):
			raise Exception(f"Method {self.method} cannot be used with quadratic constraints")

		if(self.method=='DC3' and (cs.has_soc_constraints or cs.has_sdp_constraints)):
			raise NotImplementedError

		if(self.method=='DC3'):
			assert args_DC3 is not None
			self.args_DC3=args_DC3

		self.k=cs.k #Dimension of the ambient space
		self.n=cs.n #Dimension of the embedded space

		D=cs.A_p/((cs.b_p-cs.A_p@cs.z0)@np.ones((1,cs.n)))
			
		all_P, all_q, all_r = utils.getAllPqrFromQcs(cs.qcs)
		all_M, all_s, all_c, all_d= utils.getAllMscdFromQcs(cs.socs)

		if(cs.has_sdp_constraints):
			all_F=copy.deepcopy(cs.sdpc.all_F)
			H=all_F[-1]
			for i in range(cs.sdpc.dim()):
				H += cs.y0[i,0]*cs.sdpc.all_F[i]
			Hinv=np.linalg.inv(H);
			self.register_buffer("Hinv", torch.Tensor(Hinv))
		else:
			all_F=[]

		#See https://discuss.pytorch.org/t/model-cuda-does-not-convert-all-variables-to-cuda/114733/9
		# and https://discuss.pytorch.org/t/keeping-constant-value-in-module-on-correct-device/10129
		self.register_buffer("D", torch.tensor(D))
		self.register_buffer("all_P", torch.Tensor(np.array(all_P)))
		self.register_buffer("all_q", torch.Tensor(np.array(all_q)))
		self.register_buffer("all_r", torch.Tensor(np.array(all_r)))
		self.register_buffer("all_M", torch.Tensor(np.array(all_M)))
		self.register_buffer("all_s", torch.Tensor(np.array(all_s)))
		self.register_buffer("all_c", torch.Tensor(np.array(all_c)))
		self.register_buffer("all_d", torch.Tensor(np.array(all_d)))
		self.register_buffer("all_F", torch.Tensor(np.array(all_F)))
		self.register_buffer("A_p", torch.tensor(cs.A_p))
		self.register_buffer("b_p", torch.tensor(cs.b_p))
		self.register_buffer("y1", torch.tensor(cs.y1))
		self.register_buffer("NA_E", torch.tensor(cs.NA_E))
		self.register_buffer("z0", torch.tensor(cs.z0))
		self.register_buffer("y0", torch.tensor(cs.y0))

		if(self.method=='PP' or self.method=='UP'):
			#Section 8.1.1 of https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
			self.z_projected = cp.Variable((cs.n,1))         #projected point
			self.z_to_be_projected = cp.Parameter((cs.n,1))  #original point
			constraints= cs.getConstraintsInSubspaceCvxpy(self.z_projected)
			objective = cp.Minimize(cp.sum_squares(self.z_projected - self.z_to_be_projected))
			self.prob_projection = cp.Problem(objective, constraints)

			assert self.prob_projection.is_dpp()
			self.proj_layer = CvxpyLayer(self.prob_projection, parameters=[self.z_to_be_projected], variables=[self.z_projected])

		if(self.method=='Bar'):
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

		if(self.method=='DC3'):

			A2_DC3, b2_DC3=utils.removeRedundantEquationsFromEqualitySystem(cs.A_E, cs.b_E) 

			self.register_buffer("A2_DC3", torch.tensor(A2_DC3))
			self.register_buffer("b2_DC3", torch.tensor(b2_DC3))
			self.register_buffer("A1_DC3", torch.tensor(cs.A_I))
			self.register_buffer("b1_DC3", torch.tensor(cs.b_I))

			#Constraints are now 
			# A2_DC3 y = b2_DC3
			# A1_DC3 y <= b1_DC3

			det = 0
			i = 0
			self.neq_DC3 = self.A2_DC3.shape[0]

			#######################################################
			#Note: This section follows the original implementation of DC3: https://github.com/locuslab/DC3/blob/35437af7f22390e4ed032d9eef90cc525764d26f/utils.py#L67
			#There are probably more efficient ways to do this though (probably using the QR decomposition)
			print("DC3: Obtaining partial_vars and other_vars")
			while True:
				self.partial_vars = np.random.choice(self.k, self.k - self.neq_DC3, replace=False)
				self.other_vars = np.setdiff1d( np.arange(self.k), self.partial_vars)
				det = torch.det(self.A2_DC3[:, self.other_vars])
				if(abs(det) > 0.0001):
					break
				i += 1
				if i > 1e8:
					raise Exception
			print("Done")
			#######################################################


			A2p = self.A2_DC3[:, self.partial_vars]
			A2oi = torch.inverse(self.A2_DC3[:, self.other_vars])			


			####################################################
			####################################################

			A1p=self.A1_DC3[:, self.partial_vars];
			A1o=self.A1_DC3[:, self.other_vars];

			A1_effective = A1p - A1o @ (A2oi @ A2p)
			b1_effective = self.b1_DC3 - A1o @ A2oi @ self.b2_DC3;

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

				b2=self.b2_DC3

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
		elif(self.method=='UU'):
			self.forwardForMethod=self.forwardForUU
			self.dim_after_map=self.k
		elif(self.method=='Bar'):
			self.forwardForMethod=self.forwardForBar
			self.dim_after_map=(self.num_vertices + self.num_rays)
		elif(self.method=='PP'):
			self.forwardForMethod=self.forwardForPP
			self.dim_after_map=(self.n)
		elif(self.method=='UP'):
			self.forwardForMethod=self.forwardForUP
			self.dim_after_map=(self.n)
		elif(self.method=='DC3'):
			self.forwardForMethod=self.forwardForDC3
			self.dim_after_map=(self.k - self.neq_DC3)
		else:
			raise NotImplementedError

		if(create_map):
			assert input_dim is not None, "input_dim needs to be provided"
			self.mapper=nn.Linear(input_dim, self.dim_after_map);
		else:
			self.mapper=nn.Sequential(); #Mapper does nothing

	def obtainyoFromypDC3(self, yp):
		return self.A2oi @ (self.b2_DC3 - self.A2p @ yp)


	def forwardForDC3(self, q):
		
		#### Complete partial
		y = torch.zeros((q.shape[0], self.k, 1), device=q.device)
		y[:, self.partial_vars, :] = q
		y[:, self.other_vars, :] = self.obtainyoFromypDC3(q)

		#### Grad steps all

		y_new = y
		step_index = 0
		old_y_step = 0

		if(self.training):
			max_steps=self.args_DC3['max_steps_training'] #This is called corrTrainSteps in DC3 original code
		else:
			max_steps=self.args_DC3['max_steps_testing'] #float("inf") #This is called corrTestMaxSteps in DC3 original code

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
			
			new_y_step = self.args_DC3['lr'] * y_step + self.args_DC3['momentum'] * old_y_step
			y_new = y_new - new_y_step
			old_y_step = new_y_step
			step_index += 1

			################################################
			################################################ COMPUTE current violation
			stacked=self.A1_DC3@y_new - self.b1_DC3
			for i in range(self.all_P.shape[0]): #for each of the quadratic constraints
				stacked=torch.cat((stacked,utils.quadExpression(y_new, self.all_P[i,:,:], self.all_q[i,:,:], self.all_r[i,:,:])), dim=1)
			violation=torch.max(torch.relu(stacked))
			################################################
			################################################

			# print(f"step_index={step_index}, Violation={violation}")

			converged_ineq = (violation < self.args_DC3['eps_converge'])
			max_iter_reached = (step_index >= max_steps)

			if(max_iter_reached):
				# utils.printInBoldRed("Max iter reached")
				break

			if(converged_ineq):
				# utils.printInBoldRed("Converged ineq reached")

				break

		return y_new


	def solveSecondOrderEq(self, a, b, c, is_quad_constraint):
		discriminant = torch.square(b) - 4*(a)*(c)
		assert torch.all(discriminant >= 0) 
		sol1=torch.div(  -(b)  - torch.sqrt(discriminant) , 2*a)  #note that for quad constraints the positive solution has the minus: (... - sqrt(...))/(...)
		if(is_quad_constraint):
			return sol1
		else:
			sol2=torch.div(  -(b)  +  torch.sqrt(discriminant) , 2*a) 
			return torch.relu(torch.maximum(sol1, sol2))


	def computeKappa(self,v_bar):

		kappa=torch.relu( torch.max(self.D@v_bar, dim=1, keepdim=True).values  )

		if(len(self.all_P)>0 or len(self.all_M)>0 or len(self.all_F)>0):
			rho = self.NA_E@v_bar
			rhoT=torch.transpose(rho,dim0=1, dim1=2)
			all_kappas_positives=torch.empty((v_bar.shape[0],0,1), device=v_bar.device)

			for i in range(self.all_P.shape[0]): #for each of the quadratic constraints
				P=self.all_P[i,:,:]
				q=self.all_q[i,:,:]
				r=self.all_r[i,:,:]
				
				c_prime=0.5*rhoT@P@rho;
				b_prime=(self.y0.T@P@rho + q.T@rho);
				a_prime=(0.5*self.y0.T@P@self.y0 + q.T@self.y0 +r) #This could be computed offline

				kappa_positive_i=self.solveSecondOrderEq(a_prime, b_prime, c_prime, True) 
				assert torch.all(kappa_positive_i >= 0) #If not, then either the feasible set is infeasible (note that z0 is inside the feasible set)
				all_kappas_positives = torch.cat((all_kappas_positives, kappa_positive_i), dim=1)

			for i in range(self.all_M.shape[0]): #for each of the SOC constraints
				M=self.all_M[i,:,:]
				s=self.all_s[i,:,:]
				c=self.all_c[i,:,:]
				d=self.all_d[i,:,:]

				beta=M@self.y0+s
				tau=c.T@self.y0+d

				c_prime=rhoT@M.T@M@rho - torch.square(c.T@rho)
				b_prime=2*rhoT@M.T@beta - 2*(c.T@rho)@tau
				a_prime=beta.T@beta - torch.square(tau)

				kappa_positive_i=self.solveSecondOrderEq(a_prime, b_prime, c_prime, False)

				assert torch.all(kappa_positive_i >= 0) #If not, then either the feasible set is infeasible (note that z0 is inside the feasible set)
				all_kappas_positives = torch.cat((all_kappas_positives, kappa_positive_i), dim=1)

			if(len(self.all_F)>0): #If there are SDP constraints:

				M=self.all_F[0,:,:]*rho[:,0:(0+1),0:1]
				for i in range(1,len(self.all_F)-1):
					M += self.all_F[i,:,:]*rho[:,i:(i+1),0:1] #See https://discuss.pytorch.org/t/scalar-matrix-multiplication-for-a-tensor-and-an-array-of-scalars/100174/2			

				eigenvalues = torch.unsqueeze(torch.linalg.eigvals(-self.Hinv@M),2) #Note that Hinv@M is not symmetric but always have real eigenvalues

				assert (torch.all(torch.isreal(eigenvalues)))

				kappa_positive_i = torch.relu( torch.max(eigenvalues.real, dim=1, keepdim=True).values  )
				all_kappas_positives = torch.cat((all_kappas_positives, kappa_positive_i), dim=1)


			kappa_nonlinear_constraints=(torch.max(all_kappas_positives, dim=1, keepdim=True).values)
			kappa = torch.maximum(kappa, kappa_nonlinear_constraints)


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

	def forwardForUU(self, q):
		return q

	def forwardForBar(self, q):
		tmp1 = q[:,  0:self.num_vertices,0:1] #0:1 to keep the dimension. 
		tmp2 = q[:,  self.num_vertices:(self.num_vertices+self.num_rays),0:1] #0:1 to keep the dimension. 
		
		lambdas=nn.functional.softmax(tmp1, dim=1)
		mus=torch.abs(tmp2)

		return self.getyFromz(self.V@lambdas + self.R@mus)

	def project(self, q):
		#If you use ECOS, remember to set solver_args={'eps': 1e-6} (or smaller) for better solutions, see https://github.com/cvxpy/cvxpy/issues/880#issuecomment-557278620
		z, = self.proj_layer(q, solver_args={'solve_method':'ECOS'}) #Supported: ECOS (fast, accurate), SCS (slower, less accurate).   NOT supported: GUROBI
		return z

	def forwardForPP(self, q):
		z=self.project(q)
		return self.getyFromz(z)


	def forwardForUP(self, q):
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
		#nsib denotes the number of samples in the batch
		# x has dimensions [nsib, numel_input_mapper, 1]. nsib is the number of samples in the batch (i.e., x.shape[0]=x.shape[0])
		q = self.mapper(x.view(x.size(0), -1)) #After this, q has dimensions [nsib, numel_output_mapper]
		q = torch.unsqueeze(q,dim=2)  #After this, q has dimensions [nsib, numel_output_mapper, 1]
		####################################################

		y=self.forwardForMethod(q)

		assert (torch.isnan(y).any())==False, "If you are using DC3, try reducing args_DC3[lr]"

		return y