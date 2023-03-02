import torch
import utils

class LinearConstraintWalker(torch.nn.Module):
	def __init__(self, A_np, b_np, num_steps):
		super().__init__()

		self.dim=A_np.shape[1]
		self.num_steps=num_steps;
		B_np, x0_np = utils.largestEllipsoidBInPolytope(A_np,b_np)
		self.A = torch.Tensor(A_np)
		self.b = torch.Tensor(b_np)

		self.B = torch.Tensor(B_np)
		self.x0 = torch.Tensor(x0_np)
	
		self.num_var_per_step=self.dim + 1 #3 for the direction and 1 for the length.
		self.input_numel=self.num_var_per_step*self.num_steps;

		# print(f"A = {self.A}")
		# print(f"b = {self.b}")
		# print(f"B = {self.B}")
		# print(f"x0 = {self.x0}")
		
		self.all_B=[self.B]
		self.all_x0=[self.x0]

	def plotAllSteps(self,ax):
		for i in range(len(self.all_x0)):
			B=self.all_B[i].numpy()
			x0=self.all_x0[i].numpy()
			plot2DEllipsoidB(B,x0,ax)
			ax.scatter(x0[0,0], x0[1,0])

	def forward(self, x):

		# print(x)

		B_last=self.B
		x0_last=self.x0

		self.all_B=[B_last]
		self.all_x0=[x0_last]

		for i in range(self.num_steps):
			init=(self.dim+1)*i
			v=torch.unsqueeze(x[init:(init+self.dim),0],1)
			tmp=torch.unsqueeze(x[(init+self.dim):(init+self.dim+1),0],1)
			beta=torch.sigmoid(tmp)
			# print(f"v is {v}")
			# print(f"tmp is {tmp}")
			# print(f"beta is {beta}")
			u=torch.nn.functional.normalize(v, dim=0);
			x0_last = B_last@(beta*u) + x0_last;
			#B_last=scaleEllipsoidB(B_last,self.A,self.b,x0_last)
			B_last=utils.scaleEllipsoidB(self.B,self.A,self.b,x0_last) #Note that here, self.B (instead of B_last) is preferred for numerical stability 
			self.all_B.append(B_last)
			self.all_x0.append(x0_last)

		return x0_last