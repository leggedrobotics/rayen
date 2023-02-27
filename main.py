import cvxpy as cp
import numpy as np
import torch


# E representation --> {x s.t. (x-x0)'E(x-x0) <= 1}. Here, E is a psd matrix
# B representation --> {x s.t. x=B*p_bar + x0, ||p_bar||<=1} \equiv {x s.t. ||inv(B)(x-x0)||<=1} \equiv {x s.t. (x-x0)'*inv(B)'*inv(B)*(x-x0)<=1}. 
# B is \in R^nxn (although Boyd's book says we can assume B is psd (and therefore also symmetric) without loss of generality, see section 8.4.1
# More info about the B representation: https://ieeexplore.ieee.org/abstract/document/7839930

#It returns the ellipsoid in B representation 
def largestEllipsoidBInPolytope(A,b):
	n=A.shape[1];
	B = cp.Variable((n,n), symmetric=True)
	x0 = cp.Variable((n,1))
	constraints=[]

	#Eq. 8.15 of https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
	#Also http://web.cvxr.com/cvx/examples/cvxbook/Ch08_geometric_probs/html/max_vol_ellip_in_polyhedra.html
	for i in range(A.shape[0]):
		a_i=A[i,:].T
		constraints.append(cp.norm(B@a_i)+a_i.T@x0<=b[i,0])

	objective = cp.Minimize(-cp.atoms.log_det(B))
	prob = cp.Problem(objective, constraints)
	result = prob.solve(verbose=False);
	return B.value, x0.value

def makeColumnVector(a):
	return a[:,None]

def scaleEllipsoidB(B,A,b,x0):
	minimum_so_far=torch.Tensor([float("inf")])
	for i in range(torch.numel(b)):
		# print(A[i,:])
		a_i=makeColumnVector(A[i,:])
		# print(f"a_i is{a_i}")
		tmp=(b[i,0]-a_i.mT@x0)**2/(a_i.mT@B@B.T@a_i);
		# print(f"tmp is {tmp}")
		# print(f"tmp[0,0] is {tmp[0]}")
		minimum_so_far=torch.minimum(minimum_so_far, tmp[0,0])

	print(f"-------> minimum so far={minimum_so_far}")

	return B*torch.sqrt(minimum_so_far);

class MyLayer(torch.nn.Module):
    def __init__(self, A_np, b_np):
        super().__init__()

        self.dim=A_np.shape[1]
        self.num_steps=3;
        B_np, x0_np = largestEllipsoidBInPolytope(A_np,b_np)
        self.A = torch.Tensor(A_np)
        self.b = torch.Tensor(b_np)

        self.B = torch.Tensor(B_np)
        self.x0 = torch.Tensor(x0_np)

        # print(f"A = {self.A}")
        # print(f"b = {self.b}")
        # print(f"B = {self.B}")
        # print(f"x0 = {self.x0}")

    def forward(self, x):

        print(x)

        B_last=self.B
        x0_last=self.x0

        for i in range(self.num_steps):
        	u=x[self.dim*i:self.dim*(i+1)]
        	u=makeColumnVector(u)
        	u=torch.nn.functional.normalize(u, dim=0);
        	x0_last = B_last@u + x0_last;
        	B_last=scaleEllipsoidB(B_last,self.A,self.b,x0_last)

        return u

A=np.array([[-1,0],
			 [0, -1],
			 [0, 1],
			 [0.2425,    0.9701]]);

b=np.array([[0],
			[0],
			[1],
			[1.2127]])



B, x0 = largestEllipsoidBInPolytope(A,b)

print(f"Largest ellipsoid as B={B} and x0={x0}")

x0=torch.tensor(np.array([[0.5],[0.2]]));
B=scaleEllipsoidB(torch.tensor(B),torch.tensor(A),torch.tensor(b),x0)

print(f"Using x0={x0} and scaling gives B={B}")

# print(B)
# print(x0)
# a=np.array([[4], [8], [9], [7], [2], [8]]);
# my_layer=MyLayer(A,b)
# my_layer.forward(torch.Tensor(a))