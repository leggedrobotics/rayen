import torch
import torch.nn as nn
import utils
import numpy as np
import scipy
import cvxpy as cp
import math

def checkAndGetDimAmbientSpace(Aineq, bineq, Aeq, beq):
	has_eq_constraints=((Aeq is not None) and (beq is not None));
	has_ineq_constraints=((Aineq is not None) and (bineq is not None));

	if(has_ineq_constraints):
		assert(Aineq.ndim == bineq.ndim == 2, f"Aineq.shape={Aineq.shape}, bineq.shape={bineq.shape}")
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

	return has_ineq_constraints, has_eq_constraints, dim_ambient_space 	

class Preprocessor():
	def __init__(self, Aineq, bineq, Aeq, beq):

		has_ineq_constraints, has_eq_constraints, dim_ambient_space = checkAndGetDimAmbientSpace(Aineq, bineq, Aeq, beq);

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

		utils.printInBoldBlue("-- point1")
		print(f"A=\n{A}\n b=\n{b}")

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

		utils.printInBoldBlue("-- point2")
		print(f"A=\n{A}\n b=\n{b}")

		#Find equality set
		################################################
		index_eq_set=[]

		print(f"A=\n{A}")
		print(f"b=\n{b}")

		for i in range(A.shape[0]):
			print(f"i={i}")
			objective = cp.Minimize(A[i,:]@z-b[i,0]) #I try to go far from the constraint, into the feasible set
			constraints=[A@z<=b]
			prob = cp.Problem(objective, constraints)
			result = prob.solve(verbose=False);
			obj_value=objective.value;

			if(prob.status=='unbounded'):
				obj_value=-math.inf #note that we are minimizing

			if(prob.status != 'optimal' and prob.status!='unbounded'):
				raise Exception(f"prob.status={prob.status}")

			assert(obj_value<TOL)#The objective should be negative

			if (obj_value>-TOL): #if the objective value is zero (I tried to go far from the constraint, but I couldn't)
				index_eq_set.append(i)

		utils.printInBoldGreen(f"index_eq_set={index_eq_set}")

		if(index_eq_set): #Note that in Python, empty lists are false, see https://stackoverflow.com/a/53522/6057617
			A_eq_set=A[index_eq_set,:];
			b_eq_set=b[index_eq_set,:];

			A = np.delete(A, index_eq_set, axis=0)
			b = np.delete(b, index_eq_set, axis=0)

			#At this point, if A.size==0, then it all the constraints were (hidden) equality contraitns

			#Project into the nullspace of A_eq_set
			################################################
			self.NA_eq_set=scipy.linalg.null_space(A_eq_set);
			self.p0=np.linalg.pinv(A_eq_set)@b_eq_set

			print(f"self.NA_eq_set={self.NA_eq_set}")
			print(f"self.p0={self.p0}")

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

		#Check whether or not the polyhedron is bounded
		###############################################
		# See https://github.com/TobiaMarcucci/pympc/blob/master/pympc/geometry/polyhedron.py#L529
		# and https://math.stackexchange.com/a/3593310/564801
		bounded=True
		NA_projected=scipy.linalg.null_space(self.A_projected);
		if(NA_projected.size!=0): #if the null(A_projected)!={0}
			bounded=False
		else: #if the null(A_projected)=={0} (i.e., if A_projected is invertible) 
			n0=self.A_projected.shape[0];
			n1=self.A_projected.shape[1];
			y = cp.Variable((n0,1))
			objective = cp.Minimize(cp.norm1(y))
			constraints=[self.A_projected.T@y==np.zeros((n1,1)),  y>=np.ones((n0,1))]
			prob = cp.Problem(objective, constraints)
			result = prob.solve(verbose=False);
			if(prob.status == 'infeasible'):
				bounded=False  

		if(bounded):
			utils.printInBoldBlue("Bounded feasible set")
		else:
			utils.printInBoldGreen("Unbounded feasible set")


		if(bounded):
			max_radius=None
		else:
			max_radius=1.0; #to prevent unbounded result

		self.B, self.x0 = utils.largestBallInPolytope(self.A_projected,self.b_projected, max_radius) 
		#B, x0 = utils.largestEllipsoidBInPolytope(A,b) #This is very slow in high dimensions


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

		b_minus_Ax0=torch.sub(torch.unsqueeze(self.b,dim=0),self.A@self.x0)
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
