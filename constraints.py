import utils
import cvxpy as cp
import numpy as np
from tqdm import tqdm
import scipy
import math


################### CONSTRAINTS
#everything is numpy

class LinearConstraint():
	#Constraint is A1<=b1, A2=b2
	def __init__(self, A1, b1, A2, b2):
		self.A1 = A1
		self.b1 = b1
		self.A2 = A2
		self.b2 = b2

		assert self.hasEqConstraints() or self.hasIneqConstraints()

		if (self.hasIneqConstraints()):
			assert A1.ndim == 2, f"A1.shape={A1.shape}"
			assert b1.ndim == 2, f"b1.shape={b1.shape}"
			assert b1.shape[1] ==1
			assert A1.shape[0] == b1.shape[0]

		if (self.hasEqConstraints()):
			assert A2.ndim == 2, f"A2.shape={A2.shape}"
			assert b2.ndim == 2, f"b2.shape={b2.shape}"
			assert b2.shape[1] ==1
			assert A2.shape[0] == b2.shape[0]

		if (self.hasIneqConstraints() and self.hasEqConstraints()):
			assert A1.shape[1] == A2.shape[1]

	def hasEqConstraints(self):
		return (self.A2 is not None and self.b2 is not None)

	def hasIneqConstraints(self): 
		return (self.A1 is not None and self.b1 is not None)

	def dim(self):
		if (self.A1 is not None and self.b1 is not None):
			return self.A1.shape[1]
		else:
			return self.A2.shape[1]

	def asCvxpy(self, y, epsilon=0.0):
		constraints=[];
		if self.hasIneqConstraints():
			constraints.append(self.A1@y<=self.b1)
		if self.hasEqConstraints():
			constraints.append(self.A2@y==self.b2)

		return constraints

class convexQuadraticConstraint():
	# Constraint is (1/2)x'Px + q'x +r <=0
	def __init__(self, P, q, r):
		self.P=P;
		self.q=q;
		self.r=r;

		utils.checkMatrixisPsd(self.P, tol=1e-7);
		utils.checkMatrixisNotZero(self.P);

	def dim(self):
		return self.P.shape[1]

	def asCvxpy(self, y, epsilon=0.0):

		return [0.5*cp.quad_form(y, self.P, assume_PSD=True) + self.q.T@y + self.r <= -epsilon]  #assume_PSD needs to be True because of this: https://github.com/cvxpy/cvxpy/issues/407. We have already checked that it is Psd within a tolerance

class SOCConstraint():
	#Constraint is ||My+s||<=c'y+d
	def __init__(self, M, s, c, d):
		utils.checkMatrixisNotZero(M);
		utils.checkMatrixisNotZero(c);

		assert M.shape[1]==c.shape[0]
		assert M.shape[0]==s.shape[0]
		assert s.shape[1]==1
		assert c.shape[1]==1
		assert d.shape[0]==1
		assert d.shape[1]==1

		self.M = M
		self.s = s
		self.c = c
		self.d = d

	def dim(self):
		return self.M.shape[1]

	def asCvxpy(self, y, epsilon=0.0):
		return [cp.norm(self.M@y + self.s) - self.c.T@y - self.d <= -epsilon]


class SDPConstraint():
	#Constraint is y0 F0 + y1 F1 + ... + ykm1 Fkm1 + Fk >=0
	def __init__(self, all_F):
		for F in all_F:
			utils.checkMatrixisSymmetric(F);
		self.all_F=all_F

	def dim(self):
		return (len(self.all_F)-1)

	def asCvxpy(self, y, epsilon=0.0):
		sdp_left_hand_side=0;
		k=self.dim()
		tmp=self.all_F[0].shape[0]
		for i in range(k):
			sdp_left_hand_side += y[i,0]*self.all_F[i]
		sdp_left_hand_side += self.all_F[k]

		return [sdp_left_hand_side  >>  epsilon*np.eye(tmp)]

######################################

class convexConstraints():
	def __init__(self, lc=None, qcs=[], socs=[], sdpc=None):

		if(lc is not None):
			self.has_linear_eq_constraints=lc.hasEqConstraints();
			self.has_linear_ineq_constraints=lc.hasIneqConstraints();
			self.has_linear_constraints=self.has_linear_eq_constraints or self.has_linear_ineq_constraints;
		else:
			self.has_linear_eq_constraints=False
			self.has_linear_ineq_constraints=False
			self.has_linear_constraints=False


		self.has_quadratic_constraints=(len(qcs)>0)
		self.has_soc_constraints=(len(socs)>0)
		self.has_sdp_constraints=(sdpc is not None)

		self.lc=lc
		self.qcs=qcs
		self.socs=socs
		self.sdpc=sdpc


		assert (self.has_quadratic_constraints or self.has_linear_constraints or self.has_soc_constraints or self.has_sdp_constraints), "There are no constraints!"


		#Check that the dimensions of all the constraints are the same
		all_dim=[]
		if(self.has_linear_constraints):
			all_dim.append(lc.dim())
		for qc in qcs:
			all_dim.append(qc.dim())
		for soc in socs:
			all_dim.append(soc.dim())
		if(self.has_sdp_constraints):
			all_dim.append(sdpc.dim())

		assert utils.all_equal(all_dim)
		#####################################3

		self.k=all_dim[0]


		###########################################
		# Ensure that the feasible set is not empty
		z = cp.Variable((self.k,1))
		objective = cp.Minimize(0.0)
		constraints_cvxpy = self.getConstraintsCvxpy(z)
		prob = cp.Problem(objective, constraints_cvxpy)
		result = prob.solve(verbose=False);
		if(prob.status != 'optimal'):
			raise Exception("The feasible set is empty")
		############################################

		if(self.has_linear_constraints):

			######################################## Stack the matrices so that the linear constraints look like Ax<=b 
			if(self.has_linear_ineq_constraints):
				A=self.lc.A1;
				b=self.lc.b1;
				if(self.has_linear_eq_constraints): 
					#Add the equality constraints as inequality constraints
					A=np.concatenate((A,self.lc.A2,-self.lc.A2), axis=0);
					b=np.concatenate((b,self.lc.b2,-self.lc.b2), axis=0);				
			else:
				#Add the equality constraints as inequality constraints
				A=np.concatenate((self.lc.A2,-self.lc.A2), axis=0);
				b=np.concatenate((self.lc.b2,-self.lc.b2), axis=0);
			utils.printInBoldGreen(f"A is {A.shape} and b is {b.shape}")
			########################################

			TOL=1e-7;
			z = cp.Variable((self.k,1))
			if(A.shape[0]>1): #If there is more than one constraint
				#Remove redundant constraints
				################################################
				#Eq 1.5 of https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/167108/1/thesisFinal_MaySzedlak.pdf
				#See also https://mathoverflow.net/a/69667
				utils.printInBoldBlue("Removing redundant constraints...")
				indexes_const_removed=[]
				reversed_indexes=list(reversed(range(A.shape[0])));
				for i in tqdm(reversed_indexes):
					# print(i)
					all_rows_but_i=[x for x in range(A.shape[0]) if x != i]
					objective = cp.Maximize(A[i,:]@z)
					constraints=[A[all_rows_but_i,:]@z<=b[all_rows_but_i,:],   A[i,:]@z<=(b[i,0]+1)]
					prob = cp.Problem(objective, constraints)
					result = prob.solve(verbose=False);
					if(prob.status != 'optimal' and prob.status!='optimal_inaccurate'):
						raise Exception("Value is not optimal")

					if ((objective.value-b[i,0])<=TOL):
						# print(f"Deleting constraint {i}")
						indexes_const_removed.append(i)
						A = np.delete(A, (i), axis=0)
						b = np.delete(b, (i), axis=0)

				# printInBoldBlue(f"Removed constraints {indexes_const_removed}")
				utils.printInBoldBlue(f"Removed {len(indexes_const_removed)} constraints ")
				utils.printInBoldGreen(f"A is {A.shape} and b is {b.shape}")
				################################################



			#Find equality set
			################################################
			# Section 5.2 of https://www.researchgate.net/publication/268373838_Polyhedral_Tools_for_Control
			# See also Definition 2.16 of https://sites.math.washington.edu/~thomas/teaching/m583_s2008_web/main.pdf

			E=[] #contains the indexes of the constraints in the equality set

			utils.printInBoldBlue("Finding Affine Hull and projecting...")

			for i in tqdm(range(A.shape[0])):
				objective = cp.Minimize(A[i,:]@z-b[i,0]) #I try to go far from the constraint, into the feasible set
				constraints=[A@z<=b]
				prob = cp.Problem(objective, constraints)
				result = prob.solve(verbose=False);
				obj_value=objective.value;

				if(prob.status=='unbounded'):
					obj_value=-math.inf #note that we are minimizing

				if(prob.status != 'optimal' and prob.status!='unbounded' and prob.status!='optimal_inaccurate'):
					raise Exception(f"prob.status={prob.status}")

				assert obj_value<TOL#The objective should be negative

				if (obj_value>-TOL): #if the objective value is zero (I tried to go far from the constraint, but I couldn't)
					E.append(i)

			utils.printInBoldGreen(f"E={E}")

			I=[i for i in range(A.shape[0]) if i not in E];


			#Obtain A_E, b_E and A_I, b_I
			if(len(E)>0):
				A_E=A[E,:];
				b_E=b[E,:];
			else:
				A_E=np.zeros((1,A.shape[1]));
				b_E=np.zeros((1,1));	

			if(len(I)>0):
				A_I=A[I,:];
				b_I=b[I,:];
			else:
				A_I=np.zeros((1,A.shape[1])); # 0z<=1
				b_I=np.ones((1,1));	

			#At this point, A_E, b_E, A_I, and b_I have at least one row

			#Project into the nullspace of A_E
			################################################
			NA_E=scipy.linalg.null_space(A_E);
			# n=NA_E.shape[1] #dimension of the subspace
			y1=np.linalg.pinv(A_E)@b_E
			A_p=A_I@NA_E;
			b_p=b_I-A_I@y1
					

			# print(f"A_p=\n{A_p}")
			# print(f"b_p=\n{b_p}")

			assert A_p.ndim == 2, f"A_p.shape={A_p.shape}"
			assert b_p.ndim == 2, f"b_p.shape={b_p.shape}"
			assert b_p.shape[1] ==1
			assert A_p.shape[0] == b_p.shape[0]

			# print(f"A=\n{A}")
			# print(f"b=\n{b}")

			utils.printInBoldGreen(f"A_p is {A_p.shape} and b_p is {b_p.shape}")

			self.n=A_p.shape[1] #dimension of the linear subspace

		else:
			self.n=self.k
			NA_E=np.eye(self.n);
			y1=np.zeros((self.n,1));
			A_p=np.zeros((1,self.n)) # 0z<=1
			b_p=np.ones((1,1))

			A_E=np.zeros((1,self.k)); # 0y=0
			b_E=np.zeros((1,1));	

			A_I=np.zeros((1,self.k)); # 0y<=1
			b_I=np.ones((1,1));	


		self.A_E=A_E
		self.b_E=b_E
		self.A_I=A_I
		self.b_I=b_I

		self.A_p=A_p	
		self.b_p=b_p	
		self.y1=y1	
		self.NA_E=NA_E	

		#############Obtain a strictly feasible point z0
		###################################################

		epsilon=cp.Variable()
		z0 = cp.Variable((self.n,1))

		constraints=self.getConstraintsInSubspaceCvxpy(z0, epsilon)

		constraints.append(epsilon>=0)
		constraints.append(epsilon<=0.5) #This constraint is needed for the case where the set is unbounded. Any positive value is valid
		
		objective = cp.Minimize(-epsilon)
		prob = cp.Problem(objective, constraints)

		result = prob.solve(verbose=False);
		if(prob.status != 'optimal' and prob.status!='optimal_inaccurate'):
			raise Exception(f"Value is not optimal, prob_status={prob.status}")

		assert epsilon.value>1e-8 #If not, there are no strictly feasible points in the subspace
								  #TODO: change hand-coded tolerance

		self.z0 = z0.value

		self.y0 = self.NA_E@self.z0 + self.y1	

		assert np.allclose(NA_E.T@NA_E, np.eye(NA_E.shape[1])) #By definition, N'*N=I

		###################### SET UP PROBLEM FOR PROJECTION
		###################################################
		#Section 8.1.1 of https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
		self.y_projected = cp.Variable((self.k,1))         #projected point
		self.y_to_be_projected = cp.Parameter((self.k,1))  #original point
		constraints=self.getConstraintsCvxpy(self.y_projected)
		objective = cp.Minimize(cp.sum_squares(self.y_projected - self.y_to_be_projected))
		self.prob_projection = cp.Problem(objective, constraints)
		###################################################


		##################### CHOOSE QP SOLVER FOR PROJECTION
		###################################################
		installed_solvers=cp.installed_solvers();
		if 'GUROBI' in installed_solvers:
			self.solver='GUROBI' #You need to do `python -m pip install gurobipy`
		elif 'ECOS' in installed_solvers:
			self.solver='ECOS'
		elif 'OSQP' in installed_solvers:
			self.solver='OSQP'
		elif 'CVXOPT' in installed_solvers:	
			self.solver='CVXOPT'
		else:
			#TODO: There are more solvers, see https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
			raise Exception(f"Which solver do you have installed?")
		###################################################

	def getDataAsDict(self):

		if self.has_linear_eq_constraints:
			A2=self.lc.A2;
			b2=self.lc.b2;
		else:
			A2=np.zeros((1, self.k))
			b2=np.array([[0]])			 #0==0 (i.e., no constraint)

		if self.has_linear_ineq_constraints:
			A1=self.lc.A1;
			b1=self.lc.b1;
		else:
			A1=np.zeros((1, self.k))
			b1=np.array([[1]])	          #0<=0 (i.e., no constraint)

		if self.has_quadratic_constraints:
			all_P, all_q, all_r = utils.getAllPqrFromQcs(self.qcs)	
		else:
			all_P=[np.zeros((self.k, self.k))]
			all_q=[np.zeros((self.k, 1))]
			all_r=[-np.ones((1, 1))]                    # -1<=0 (i.e., no constraint)

		if self.has_soc_constraints:
			all_M, all_s, all_c, all_d = utils.getAllMscdFromQcs(self.socs)
		else:
			all_M=[np.zeros((self.k, self.k))]
			all_s=[np.zeros((self.k, 1))]
			all_c=[np.zeros((self.k, 1))]                 
			all_d=[np.ones((1, 1))]                    # 0<=1 (i.e., no constraint)

		if(self.has_sdp_constraints):
			all_F=self.sdpc.all_F
		else:
			all_F=[]
			for i in range(self.k):
				all_F.append(np.zeros((self.k, self.k)))
			all_F.append(np.eye((self.k)))            # I<=0 (i.e., no constraint)


		return dict(A2=A2, b2=b2, A1=A1, b1=b1, 
					all_P=all_P, all_q=all_q, all_r=all_r, 
					all_M=all_M, all_s=all_s, all_c=all_c, all_d=all_d,
					all_F=all_F)



	######################### CONSTRAINTS IN THE SUBSPACE
	def getConstraintsInSubspaceCvxpy(self, z, epsilon=0.0):

		constraints = [self.A_p@z - self.b_p <= -epsilon*np.ones((self.A_p.shape[0],1))]

		y=self.NA_E@z + self.y1

		constraints+=self.getNonLinearConstraintsCvxpy(y, epsilon)

		return constraints
	###########################################################################

	def getNonLinearConstraintsCvxpy(self, y, epsilon=0.0):

		constraints=[]

		for qc in self.qcs:   
			constraints += qc.asCvxpy(y, epsilon) 

		for soc in self.socs:   
			constraints += soc.asCvxpy(y, epsilon) 

		if(self.has_sdp_constraints):
			constraints += self.sdpc.asCvxpy(y, epsilon) 	

		return constraints	

	def getConstraintsCvxpy(self, y, epsilon=0.0):

		constraints=[]

		if(self.has_linear_constraints):
			constraints += self.lc.asCvxpy(y, epsilon) 

		constraints+=self.getNonLinearConstraintsCvxpy(y, epsilon)

		return constraints


	#######################################3

	def project(self, y_to_be_projected):
		assert y_to_be_projected.shape==self.y_to_be_projected.shape

		self.y_to_be_projected.value=y_to_be_projected;
		obj_value = self.prob_projection.solve(verbose=False, solver=self.solver);

		if(self.prob_projection.status != 'optimal' and self.prob_projection.status != 'optimal_inaccurate'):
			raise Exception(f"Value is not optimal, prob_status={self.prob_projection.status}")

		return self.y_projected.value, obj_value	

	def getViolation(self, y_to_be_projected):

		if(y_to_be_projected.ndim==1):
			#convert to a column vector
			y_to_be_projected=np.expand_dims(y_to_be_projected, axis=1)

		_, violation = self.project(y_to_be_projected)

		assert violation>=0  #violation is nonnegative by definition

		return violation;