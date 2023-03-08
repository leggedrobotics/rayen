import torch
import torch.nn as nn
import utils
import numpy as np
import scipy
import cvxpy as cp

class Preprocessor():
	def __init__(self, Aineq, bineq, Aeq, beq):

		has_eq_constraints=((Aeq is not None) and (beq is not None));
		has_ineq_constraints=((Aineq is not None) and (bineq is not None));

		if(has_ineq_constraints):
			assert(Aineq.ndim == bineq.ndim == 2)
			assert(bineq.shape[1] ==1)
			assert(Aineq.shape[0] == bineq.shape[0])
			dim_ambient_space=Aineq.shape[1]

		if(has_eq_constraints):
			assert(Aeq.ndim == beq.ndim == 2)
			assert(beq.shape[1] ==1)
			assert(Aeq.shape[0] == beq.shape[0])
			dim_ambient_space=Aeq.shape[1]

		if(has_eq_constraints and has_ineq_constraints):
			assert(Aineq.shape[1] == Aeq.shape[1])

		if(has_eq_constraints==False and has_ineq_constraints==False):
			raise Exception("There are no constraints!")

		z = cp.Variable((dim_ambient_space,1))

		TOL=1e-7;

		################################################
		#Make sure that the LP has a feasible solution
		objective = cp.Minimize(0.0)
		constraints=[]
		if(has_eq_constraints):
			constraints.append(Aeq@z==beq)
			print(f"Aeq.shape={Aeq.shape}")
			print(f"beq.shape={beq.shape}")
		if(has_ineq_constraints):
			print(f"Aineq.shape={Aineq.shape}")
			print(f"bineq.shape={bineq.shape}")
			constraints.append(Aineq@z<=bineq)
		prob = cp.Problem(objective, constraints)
		result = prob.solve(verbose=False);
		if(prob.status != 'optimal'):
			raise Exception("The feasible set is empty")
		################################################

		if(has_ineq_constraints):
			A=Aineq;
			b=bineq;
			if(has_eq_constraints): 
				#Add the equality constraints as inequality constraints
				A=np.concatenate((A,Aeq,-Aeq), axis=0);
				b=np.concatenate((b,beq,-beq), axis=0);				
		else:
			#Add the equality constraints as inequality constraints
			A=np.concatenate((Aeq,-Aeq), axis=0);
			b=np.concatenate((Aeq,-Aeq), axis=0);

		#At this point, the feasible region is represented by Ax<=b		

		#Remove redundant constraints
		################################################
		for i in reversed(range(A.shape[0])):
			all_rows_but_i=[x for x in range(A.shape[0]) if x != i]
			objective = cp.Maximize(A[i,:]@z)
			constraints=[A[all_rows_but_i,:]@z<=b[all_rows_but_i,:],   A[i,:]@z<=(b[i,0]+1)]
			prob = cp.Problem(objective, constraints)
			result = prob.solve(verbose=False);
			if(prob.status != 'optimal'):
				raise Exception("Value is not optimal")

			if ((objective.value-b[i,0])<=TOL):
				print(f"Deleting constraint {i}")
				A = np.delete(A, (i), axis=0)
				b = np.delete(b, (i), axis=0)

		#Find equality set
		################################################
		index_eq_set=[]
		for i in range(A.shape[0]):
			all_rows_but_i=[x for x in range(A.shape[0]) if x != i]
			objective = cp.Minimize(A[i,:]@z-b[i,0])
			constraints=[A@z<=b]
			prob = cp.Problem(objective, constraints)
			result = prob.solve(verbose=False);
			if(prob.status != 'optimal'):
				raise Exception("Value is not optimal")

			assert(objective.value<TOL)#The objective should be negative

			if (objective.value>-TOL): #if the objective value is zero
				index_eq_set.append(i)

		print(f"index_eq_set={index_eq_set}")

		if(index_eq_set): #Note that in Python, empty sequences are false, see https://stackoverflow.com/a/53522/6057617
			A_eq_set=A[index_eq_set,:];
			b_eq_set=b[index_eq_set,:];

			A = np.delete(A, index_eq_set, axis=0)
			b = np.delete(b, index_eq_set, axis=0)

			#Project into the nullspace of A_eq_set
			################################################
			self.NA_eq_set=scipy.linalg.null_space(A_eq_set);
			self.p0=np.linalg.pinv(A_eq_set)@b_eq_set
			self.A_projected=A@self.NA_eq_set;
			self.b_projected=b-A@self.p0
		else: 
			#no need to project
			self.NA_eq_set=np.eye(dim_ambient_space);
			self.p0=np.zeros((dim_ambient_space,1))
			self.A_projected=A
			self.b_projected=b
				

		print(f"A_projected=\n{self.A_projected}")
		print(f"b_projected=\n{self.b_projected}")

		print(f"A=\n{A}")
		print(f"b=\n{b}")


		#B, x0 = utils.largestEllipsoidBInPolytope(A,b) #This step is very slow in high dimensions
		self.B, self.x0 = utils.largestBallInPolytope(self.A_projected,self.b_projected) 

		# return B.value, x0.value


class LinearConstraintWalker(torch.nn.Module):
	def __init__(self, Aineq_np, bineq_np, Aeq_np, beq_np):
		super().__init__()


		pre=Preprocessor(Aineq_np, bineq_np, Aeq_np, beq_np);

		self.dim=pre.A_projected.shape[1]

		self.A = torch.Tensor(pre.A_projected)
		self.b = torch.Tensor(pre.b_projected)
		self.B = torch.Tensor(pre.B)
		self.x0 = torch.Tensor(pre.x0)
		self.NA_eq_set=torch.Tensor(pre.NA_eq_set)
		self.p0=torch.Tensor(pre.p0)

		#add batches dimension
		self.B=torch.unsqueeze(self.B,0)
		self.x0=torch.unsqueeze(self.x0,0)

		self.mapper=nn.Sequential();

	def getNumelInputWalker(self):
		return (self.dim+1)

	def setMapper(self, mapper):
		self.mapper=mapper

	def forward(self, x):

		self.x0 = self.x0.to(x.device)
		self.B = self.B.to(x.device)
		self.A = self.A.to(x.device)
		self.b = self.b.to(x.device)

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
		tmp= torch.unsqueeze(z[:, self.dim:(self.dim+1),0],2)
		beta=torch.sigmoid(tmp)
		u=torch.nn.functional.normalize(v, dim=1);

		b_minus_Ax0=torch.sub(torch.unsqueeze(self.b,dim=0),self.A@self.x0)
		all_max_distances=torch.div(b_minus_Ax0,self.A@u)
		all_max_distances[all_max_distances<=0]=float("Inf")
		#Note that we know that self.x0 is a strictly feasible point of the set
		tmp = torch.min(all_max_distances, dim=1, keepdim=True)
		max_distance =tmp.values

		#Note that something like:
		#max_distance = torch.min(all_max_distances[all_max_distances>=0], dim=1)
		#would remove the dimensions (since the mask may delete more/fewer elements per batch)

		x0_new = beta*max_distance*u + self.x0;
		
		#Now lift back to the original space
		x0_new =self.NA_eq_set@x0_new + self.p0


		return x0_new
