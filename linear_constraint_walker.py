import torch
import torch.nn as nn
import utils
import numpy as np
import scipy

# class MapperAndWalker(nn.Module):
#     def __init__(self, A_np, b_np, num_steps, numel_input_mapper):
#         super().__init__()
#         self.constraint = LinearConstraintWalker(A_np, b_np, num_steps=num_steps, use_max_ellipsoid=False)
#         print(f"numel_input_mapper={numel_input_mapper}")
#         print(f"self.constraint.getNumelInput()={self.constraint.getNumelInput()}")
#         self.net = nn.Sequential(nn.Linear(numel_input_mapper, self.constraint.getNumelInput()))

#     def forward(self, x):
#         orig_size = x.size()
#         y = x.view(, -1)
#         z = self.constraint(self.net(y))
#         z = z.view(orig_size)
#         return z

class LinearConstraintWalker(torch.nn.Module):
	def __init__(self, Aineq_np, bineq_np, Aeq_np, beq_np, num_steps, method, use_max_ellipsoid):
		super().__init__()


		self.method=method

		if np.allclose(bineq_np, 0):
			raise ValueError("Constraint set is a zero-centered cone.")

		#inequality stuff
		assert(Aineq_np.ndim == bineq_np.ndim == 2)
		assert(bineq_np.shape[1] ==1)
		assert(Aineq_np.shape[0] == bineq_np.shape[0])

		#Equality stuff
		if((Aeq_np is not None) and (beq_np is not None)):
			assert(Aeq_np.ndim == beq_np.ndim == 2)
			assert(beq_np.shape[1] ==1)
			assert(Aeq_np.shape[0] == beq_np.shape[0])

			assert(Aineq_np.shape[1] == Aeq_np.shape[1])
			self.has_ineq_constraints=True

			#TODO: check what the null_space is
			N_np=scipy.linalg.null_space(Aeq_np);
			p0_np=np.linalg.pinv(Aeq_np)@beq_np
			A_np=Aineq_np@N_np;
			b_np=bineq_np-Aineq_np@p0_np

			self.N = torch.Tensor(N_np)
			self.p0 = torch.Tensor(p0_np)

			self.dim=N_np.shape[1] #TODO: what if the polytope has no feasible point


		else:
			self.has_ineq_constraints=False
			A_np=Aineq_np
			b_np=bineq_np
			self.dim=A_np.shape[1]


		self.num_steps=num_steps;

		# print("Solving for the largest Ellipsoid...")
		if(use_max_ellipsoid==True):
			B_np, x0_np = utils.largestEllipsoidBInPolytope(A_np,b_np) #This step is very slow in high dimensions
		else:
			B_np, x0_np = utils.largestBallInPolytope(A_np,b_np) 
		# print("Solved for the largest Ellipsoid")

		# self.dim=A_np.shape[1]
		# self.num_variables_C=int(self.dim*(self.dim+1)/2) #C is a symmetric matrix such that C'*C=C*C=B. 


		self.A = torch.Tensor(A_np)
		self.b = torch.Tensor(b_np)
		self.B = torch.Tensor(B_np)
		self.x0 = torch.Tensor(x0_np)

		#add batches dimension
		self.B=torch.unsqueeze(self.B,0)
		self.x0=torch.unsqueeze(self.x0,0)

		self.num_var_per_step=self.dim + 1 #3 for the direction and 1 for the length.

		self.mapper=nn.Sequential();

	def getNumelInputWalker(self):
		if(self.method=="ellipsoid_walker"):
			return (self.dim+1)*self.num_steps
		elif(self.method=="point_walker"):
			return (self.dim+1)
		else:
			assert(False,"Method not implemented yet") 

	def setMapper(self, mapper):
		self.mapper=mapper

	def plotAllSteps(self,ax):
		batch_index=0 #for now just plot the first element of the batch
		for i in range(len(self.all_x0)):
			print(f"self.all_B[i].shape={self.all_B[i].shape}")
			B=self.all_B[i][batch_index,:,:].numpy()
			x0=self.all_x0[i][batch_index,:,:].numpy()
			utils.plot2DEllipsoidB(B,x0,ax)
			ax.scatter(x0[0,0], x0[1,0])

	def liftBack(self, q):
		print(f"self.N.shape={self.N.shape}")
		print(f"self.p0.shape={self.p0.shape}")
		print(f"q.shape={q.shape}")
		return self.N@q + self.p0

	def forward(self, x):

		self.x0 = self.x0.to(x.device)
		self.B = self.B.to(x.device)
		self.A = self.A.to(x.device)
		self.b = self.b.to(x.device)

		print(f"x.shape before={x.shape}")
		# print(f"self.mapper={self.mapper}")

		##################  MAPPER LAYER ####################
		# x has dimensions [num_batches, numel_input_mapper, 1]
		y = x.view(x.size(0), -1)
		# y has dimensions [num_batches, numel_input_mapper] This is needed to be able to pass it through the linear layer
		z = self.mapper(y)
		#Here z has dimensions [num_batches, numel_input_walker]
		z = torch.unsqueeze(z,dim=2)
		#Here z has dimensions [num_batches, numel_input_walker, 1]
		####################################################


		# print("========================In forward ========================")
		# print(f"x={x}")
		# print(f"x.shape={x.shape}")


		B_last=self.B
		x0_last=self.x0

		self.all_x0=[ x0_last  ]
		self.all_B=[ B_last  ] 

		# print(f"First x0.shape={x0_last.shape}\n")

		if(self.method=="ellipsoid_walker"):
			for i in range(self.num_steps):
				# print("========================New iteration ========================")
				init=(self.dim+1)*i #+ self.num_variables_C
				v = torch.unsqueeze(z[:,  init:(init+self.dim),0],2)             
				tmp= torch.unsqueeze(z[:, (init+self.dim):(init+self.dim+1),0],2)
				beta=torch.sigmoid(tmp)
				u=torch.nn.functional.normalize(v, dim=1);
				beta_times_u = beta*u #Using elementwise multiplication and broadcasting here
				x0_last = torch.matmul(B_last, beta_times_u) + x0_last;
				B_last=utils.scaleEllipsoidB(self.B,self.A,self.b,x0_last) #Note that here, self.B (instead of B_last) is preferred for numerical stability 
				self.all_B.append(B_last)
				self.all_x0.append(x0_last)

		elif(self.method=="point_walker"):

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


			# print(f"all_max_distances={max_distance}") 
			# print(f"max_distance={max_distance}")
			x0_last = beta*max_distance*u + self.x0;
		else:
			assert(False,"Method not implemented yet")



		#Now lift back to the original space
		if(self.has_ineq_constraints==True):
			x0_last = self.liftBack(x0_last)

		return x0_last