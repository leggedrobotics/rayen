# --------------------------------------------------------------------------
# Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
# See LICENSE file for the license information
# -------------------------------------------------------------------------- 

import torch
import torch.nn as nn
from . import utils
import numpy as np
import cvxpy as cp
import math
from cvxpylayers.torch import CvxpyLayer
import random
import copy
import time

class ConstraintModule(torch.nn.Module):
	def __init__(self, cs, input_dim=None, method='RAYEN', create_map=True, args_DC3=None):
		super().__init__()


		self.method=method

		if(self.method=='Bar' and cs.has_quadratic_constraints):
			raise Exception(f"Method {self.method} cannot be used with quadratic constraints")

		if(self.method=='DC3' and (cs.has_soc_constraints or cs.has_lmi_constraints)):
			raise NotImplementedError

		if(self.method=='DC3'):
			utils.verify(args_DC3 is not None)
			self.args_DC3=args_DC3

		self.cs=cs
		self.k=cs.k #Dimension of the ambient space
		self.n=cs.n #Dimension of the embedded space

		D=cs.A_p/((cs.b_p-cs.A_p@cs.z0)@np.ones((1,cs.n)))
			
		all_P, all_q, all_r = utils.getAllPqrFromQcs(cs.qcs)
		all_M, all_s, all_c, all_d= utils.getAllMscdFromSocs(cs.socs)

		if(cs.has_lmi_constraints):
			all_F=copy.deepcopy(cs.lmic.all_F)
			H=all_F[-1]
			for i in range(cs.lmic.dim()):
				H += cs.y0[i,0]*cs.lmic.all_F[i]
			Hinv=np.linalg.inv(H)
			mHinv=-Hinv;
			L=np.linalg.cholesky(Hinv) # Hinv = L @ L^T 
			self.register_buffer("mHinv", torch.Tensor(mHinv))
			self.register_buffer("L", torch.Tensor(L))

		else:
			all_F=[]

		#See https://discuss.pytorch.org/t/model-cuda-does-not-convert-all-variables-to-cuda/114733/9
		# and https://discuss.pytorch.org/t/keeping-constant-value-in-module-on-correct-device/10129
		self.register_buffer("D", torch.Tensor(D))
		self.register_buffer("all_P", torch.Tensor(np.array(all_P)))
		self.register_buffer("all_q", torch.Tensor(np.array(all_q)))
		self.register_buffer("all_r", torch.Tensor(np.array(all_r)))
		self.register_buffer("all_M", torch.Tensor(np.array(all_M)))
		self.register_buffer("all_s", torch.Tensor(np.array(all_s)))
		self.register_buffer("all_c", torch.Tensor(np.array(all_c)))
		self.register_buffer("all_d", torch.Tensor(np.array(all_d)))
		# self.register_buffer("all_F", torch.Tensor(np.array(all_F))) #This one dies (probably because out of memory) when all_F contains more than 7000 matrices 500x500 approx
		self.register_buffer("all_F", torch.Tensor(all_F))
		self.register_buffer("A_p", torch.Tensor(cs.A_p))
		self.register_buffer("b_p", torch.Tensor(cs.b_p))
		self.register_buffer("yp", torch.Tensor(cs.yp))
		self.register_buffer("NA_E", torch.Tensor(cs.NA_E))
		self.register_buffer("z0", torch.Tensor(cs.z0))
		self.register_buffer("y0", torch.Tensor(cs.y0))

		if(self.method=='PP' or self.method=='UP'):
			#Section 8.1.1 of https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
			self.z_projected = cp.Variable((self.n,1))         #projected point
			self.z_to_be_projected = cp.Parameter((self.n,1))  #original point
			constraints= self.cs.getConstraintsInSubspaceCvxpy(self.z_projected)

			#First option.
			objective = cp.Minimize(cp.sum_squares(self.z_projected - self.z_to_be_projected))

			#Second option. Sometimes this may be preferred because of this: http://cvxr.com/cvx/doc/advanced.html#eliminating-quadratic-forms This may solve cases of ("Solver ecos returned status Infeasible" or "Solver SCS returned status Infeasible")
			# objective = cp.Minimize(cp.norm(self.z_projected - self.z_to_be_projected))

			self.prob_projection = cp.Problem(objective, constraints)

			assert self.prob_projection.is_dpp()
			self.proj_layer = CvxpyLayer(self.prob_projection, parameters=[self.z_to_be_projected], variables=[self.z_projected])

			if(self.cs.has_lmi_constraints):
				self.solver_projection='SCS' #slower, less accurate, supports LMI constraints
			else:
				self.solver_projection='ECOS' #fast, accurate, does not support LMI constraints


		if(self.method=='RAYEN' or self.method=='RAYEN_old'):

			if(cs.has_quadratic_constraints):
				all_delta=[]
				all_phi=[]

				for i in range(self.all_P.shape[0]): #for each of the quadratic constraints
					P=self.all_P[i,:,:]
					q=self.all_q[i,:,:]
					r=self.all_r[i,:,:]
					y0=self.y0

					sigma=2*(0.5*y0.T@P@y0 + q.T@y0 + r)
					phi = -(y0.T@P + q.T)/sigma
					delta= ( (y0.T@P + q.T).T@(y0.T@P + q.T) - 4*(0.5*y0.T@P@y0 + q.T@y0 + r)*0.5*P         )/torch.square(sigma)

					all_delta.append(delta)
					all_phi.append(phi)

				all_delta = torch.stack(all_delta)
				all_phi = torch.stack(all_phi)

				self.register_buffer("all_delta", all_delta)
				self.register_buffer("all_phi", all_phi)

		if(self.method=='Bar'):
			print("Computing vertices and rays...")
			V, R = utils.H_to_V(cs.A_p, cs.b_p);
			self.register_buffer("V", torch.Tensor(V))
			self.register_buffer("R", torch.Tensor(R))
			self.num_vertices=self.V.shape[1];
			self.num_rays=self.R.shape[1];
			assert (self.num_vertices+self.num_rays)>0
			print(f"Found {self.num_vertices} vertices and {self.num_rays} rays")

		if(self.method=='DC3'):

			A2_DC3, b2_DC3=utils.removeRedundantEquationsFromEqualitySystem(cs.A_E, cs.b_E) 

			self.register_buffer("A2_DC3", torch.Tensor(A2_DC3))
			self.register_buffer("b2_DC3", torch.Tensor(b2_DC3))
			self.register_buffer("A1_DC3", torch.Tensor(cs.A_I))
			self.register_buffer("b1_DC3", torch.Tensor(cs.b_I))

			#Constraints are now 
			# A2_DC3 y = b2_DC3
			# A1_DC3 y <= b1_DC3

			self.neq_DC3 = self.A2_DC3.shape[0]

			#################################### Find partial_vars and other_vars

			if(A2_DC3.shape[0]==0): #There are no equality constraints
				self.partial_vars=np.arange(self.k)
				self.other_vars=np.setdiff1d( np.arange(self.k), self.partial_vars)
			else:
				# This is a more efficient way to do https://github.com/locuslab/DC3/blob/35437af7f22390e4ed032d9eef90cc525764d26f/utils.py#L67
				# Here, we follow  https://stackoverflow.com/a/27907936
				(A2_DC3_rref, pivots_pos, row_exchanges) = utils.rref(A2_DC3);
				self.other_vars = [i[1] for i in pivots_pos];
				self.partial_vars = np.setdiff1d( np.arange(self.k), self.other_vars)

			#######################################################

			A2p = self.A2_DC3[:, self.partial_vars]
			A2o = self.A2_DC3[:, self.other_vars]

			# assert np.linalg.matrix_rank(A2_DC3) == np.linalg.matrix_rank(A2o) == A2o.shape[-1]

			A2oi = torch.inverse(A2o)			

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
				# tmp=random.randint(1, 100) #number of elements in the batch
				# yp=torch.rand(tmp, len(self.partial_vars), 1) 

				# y = torch.zeros((tmp, self.k, 1))
				# y[:, self.partial_vars, :] = yp
				# y[:, self.other_vars, :] = self.obtainyoFromypDC3(yp)

				# using_effective=utils.quadExpression(yp, P_effective, q_effective, r_effective)
				# using_original=utils.quadExpression(y, P, q, r)

				# assert torch.allclose(using_effective, using_original, atol=1e-05) 

				###################

				all_P_effective[i,:,:]=P_effective
				all_q_effective[i,:,:]=q_effective
				all_r_effective[i,:,:]=r_effective

			self.register_buffer("all_P_effective", all_P_effective)
			self.register_buffer("all_q_effective", all_q_effective)
			self.register_buffer("all_r_effective", all_r_effective)

			####################################################
			####################################################


		if(self.method=='RAYEN_old'):
			self.forwardForMethod=self.forwardForRAYENOld
			self.dim_after_map=(self.n+1)
		elif(self.method=='RAYEN'):
			self.forwardForMethod=self.forwardForRAYEN
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
			assert (self.dim_after_map==self.n)
		else:
			raise NotImplementedError

		if(create_map):
			utils.verify(input_dim is not None, "input_dim needs to be provided")
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

			converged_ineq = (violation < self.args_DC3['eps_converge'])
			max_iter_reached = (step_index >= max_steps)

			if(max_iter_reached):
				break

			if(converged_ineq):
				break

		return y_new


	def solveSecondOrderEq(self, a, b, c, is_quad_constraint):
		discriminant = torch.square(b) - 4*(a)*(c)

		assert torch.all(discriminant >= 0), f"Smallest element is {torch.min(discriminant)}"
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

				#FIRST WAY (slower, easier to understand)
				# P=self.all_P[i,:,:]
				# q=self.all_q[i,:,:]
				# r=self.all_r[i,:,:]
				
				# c_prime=0.5*rhoT@P@rho;
				# b_prime=(self.y0.T@P+ q.T)@rho;
				# a_prime=(0.5*self.y0.T@P@self.y0 + q.T@self.y0 +r) 

				# kappa_positive_i_first_way=self.solveSecondOrderEq(a_prime, b_prime, c_prime, True) 

				#SECOND WAY (faster)
				kappa_positive_i = self.all_phi[i,:,:]@rho + torch.sqrt(rhoT@self.all_delta[i,:,:]@rho)


				# assert torch.allclose(kappa_positive_i,kappa_positive_i_first_way, atol=1e-06), f"{torch.max(torch.abs(kappa_positive_i-kappa_positive_i_first_way))}"


				assert torch.all(kappa_positive_i >= 0), f"Smallest element is {kappa_positive_i}" #If not, then Z may not be feasible (note that z0 is in the interior of Z)
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

			if(len(self.all_F)>0): #If there are LMI constraints:

				############# OBTAIN S
				# First option (much slower)
				# S=self.all_F[0,:,:]*rho[:,0:(0+1),0:1]
				# for i in range(1,len(self.all_F)-1):
				# 	#See https://discuss.pytorch.org/t/scalar-matrix-multiplication-for-a-tensor-and-an-array-of-scalars/100174/2	
				# 	S += self.all_F[i,:,:]*rho[:,i:(i+1),0:1] 


				# Second option (much faster)
				S=torch.einsum('ajk,ial->ijk', [self.all_F[0:-1,:,:], rho]) #See the tutorial https://rockt.github.io/2018/04/30/einsum
				
				############# COMPUTE THE EIGENVALUES

				## Option 1: (compute whole spectrum of the matrix, using the non-symmetric matrix self.mHinv@S)
				# eigenvalues = torch.unsqueeze(torch.linalg.eigvals(self.mHinv@S),2) #Note that mHinv@M is not symmetric but always have real eigenvalues
				# assert (torch.all(torch.isreal(eigenvalues)))
				# largest_eigenvalue = torch.max(eigenvalues.real, dim=1, keepdim=True).values 
				
				LTmSL=self.L.T @ (-S) @ self.L #This matrix is symmetric

				## Option 2: (compute whole spectrum of the matrix, using the symmetric matrix LTmSL). Much faster than Option 1
				eigenvalues = torch.unsqueeze(torch.linalg.eigvalsh(LTmSL),2) #Note that L^T (-S) L is a symmetric matrix
				largest_eigenvalue = torch.max(eigenvalues, dim=1, keepdim=True).values 

				## Option 3: Use LOBPCG with A=LTmSL and B=I. The advantage of this method is that only the largest eigenvalue is computed. But, empirically, this option is faster than option 2 only for very big matrices (>1000x1000)
				# guess_lobpcg=torch.rand(1, H.shape[0], 1);
				# size_batch=v_bar.shape[0]
				# largest_eigenvalue, _ = torch.lobpcg(A=LTmSL, k=1, B=None, niter=-1) #, X=guess_lobpcg.expand(size_batch, -1, -1)
				# largest_eigenvalue=torch.unsqueeze(largest_eigenvalue, 1)

				## Option 4: Use power iteration to compute the largest eigenvalue. Often times is slower than just computing the whole spectrum, and sometimes it does not converge
				# guess_v = torch.nn.functional.normalize(torch.rand(S.shape[1],1), dim=0)
				# largest_eigenvalue=utils.findLargestEigenvalueUsingPowerIteration(self.mHinv@S, guess_v)


				## Option 5: Use LOBPCG with A=-S and B=H. There are two problems though:
				# --> This issue: https://github.com/pytorch/pytorch/issues/101075
				# --> Backward is not implemented for B!=I, see: https://github.com/pytorch/pytorch/blob/d54fcd571af48685b0699f6ac1e31b6871d0d768/torch/_lobpcg.py#L329 

				## Option 6: Use https://github.com/rfeinman/Torch-ARPACK with LTmSL. The problem is that backward() is not implemented yet 

				## Option 7: Use https://github.com/buwantaiji/DominantSparseEigenAD. But it does not have support for batched matrices, see https://github.com/buwantaiji/DominantSparseEigenAD/issues/1

				kappa_positive_i = torch.relu( largest_eigenvalue )

				
				all_kappas_positives = torch.cat((all_kappas_positives, kappa_positive_i), dim=1)


			kappa_nonlinear_constraints=(torch.max(all_kappas_positives, dim=1, keepdim=True).values)
			kappa = torch.maximum(kappa, kappa_nonlinear_constraints)


		assert torch.all(kappa >= 0)

		return kappa

	def forwardForRAYENOld(self, q):
		v = q[:,  0:self.n,0:1]
		v_bar=torch.nn.functional.normalize(v, dim=1)
		kappa=self.computeKappa(v_bar)
		beta= q[:, self.n:(self.n+1),0:1]
		alpha=1/(torch.exp(beta) + kappa) 
		return self.getyFromz(self.z0 + alpha*v_bar)

	def forwardForRAYEN(self, q):
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
		#If you use ECOS, you can set solver_args={'eps': 1e-6} (or smaller) for better solutions, see https://github.com/cvxpy/cvxpy/issues/880#issuecomment-557278620
		z, = self.proj_layer(q, solver_args={'solve_method':self.solver_projection}) # "max_iters": 10000
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
		y=self.NA_E@z + self.yp
		return y

	def getzFromy(self, y):
		z=self.NA_E.T@(y - self.yp)
		return z

	def forward(self, x):

		##################  MAPPER LAYER ####################
		#nsib denotes the number of samples in the batch
		# x has dimensions [nsib, numel_input_mapper, 1]. nsib is the number of samples in the batch (i.e., x.shape[0]=x.shape[0])
		q = self.mapper(x.view(x.size(0), -1)) #After this, q has dimensions [nsib, numel_output_mapper]
		q = torch.unsqueeze(q,dim=2)  #After this, q has dimensions [nsib, numel_output_mapper, 1]
		####################################################

		y=self.forwardForMethod(q)

		assert (torch.isnan(y).any())==False, f"If you are using DC3, try reducing args_DC3[lr]. Right now it's {self.args_DC3['lr']}"

		return y