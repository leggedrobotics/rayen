import cvxpy as cp
import numpy as np
import torch
import cdd
from scipy.spatial import ConvexHull
import math
import sympy
#import spb  #https://sympy-plot-backends.readthedocs.io/en/latest/index.html

import matplotlib.pyplot as plt

def getVertexesRaysFromAb(A, b):
	bmA= np.concatenate([b, -A], axis=1) # See https://pycddlib.readthedocs.io/en/latest/matrix.html
	bmA_cdd = cdd.Matrix(bmA.tolist(), number_type='float')
	bmA_cdd.rep_type = cdd.RepType.INEQUALITY
	poly = cdd.Polyhedron(bmA_cdd)
	gen = poly.get_generators()
	# print(gen)
	vertices, rays = getVertexesRaysFromGenerators(gen)
	return vertices, rays 

def getVertexesRaysFromGenerators(gen):
	generators=list(gen)
	vertices=np.array([[],[]]);
	rays=np.array([[],[]]);
	for i in range(len(generators)):
		gen_i=generators[i];
		tmp=np.asarray(gen_i[1:]).reshape((-1,1));

		if(gen_i[0]==1):
			vertices=np.append(vertices,tmp, axis=1)
		else: #it is zero
			rays=np.append(rays,tmp, axis=1)

	return vertices, rays

# https://stackoverflow.com/a/69427715/6057617
def move_sympyplot_to_axes(p, ax):
    backend = p.backend(p)
    backend.ax = ax
    backend._process_series(backend.parent._series, ax, backend.parent)
    backend.ax.spines['right'].set_color('none')
    backend.ax.spines['top'].set_color('none')
    backend.ax.spines['bottom'].set_position('zero')
    plt.close(backend.fig)

def plot2DPolyhedron(A, b, ax):
	##FIRST WAY
	# npoints=300
	# d = np.linspace(-2,16,npoints)
	# x1,x2 = np.meshgrid(d,d)

	# tmp=1;
	# for i in range(A.shape[0]):
	# 	tmp=tmp & (A[i,0]*x1 + A[i,1]*x2 <=b[i,0]);

	# plt.imshow( tmp.astype(int), extent=(x1.min(),x1.max(),x2.min(),x2.max()),origin="lower", cmap="Greys", alpha = 0.3);

	vertices, rays = getVertexesRaysFromAb(A, b)
	#See example from https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
	coord = vertices.T
	hull = ConvexHull(coord)
	for simplex in hull.simplices:
		ax.plot(coord[simplex, 0], coord[simplex, 1], 'r-')


def plot2DEllipsoidB(B,x0,ax):
	x1, x2 = sympy.symbols('x1 x2')
	x=np.array([[x1],[x2]])
	try:
		B_inv=np.linalg.inv(B);
	except np.linalg.LinAlgError as err:
		print(str(err))
		return
	tmp=(x-x0).T@B_inv.T@B_inv@(x-x0)-1 #This is [[scalar]]
	expression=tmp[0,0]; 
	f=sympy.lambdify([x1,x2], expression)

	eigenvalues=np.linalg.eigvals(B)
	tmp_eig=np.amax(eigenvalues);
	xx = np.linspace(x0[0,0]-tmp_eig, x0[0,0]+tmp_eig, 600)
	yy = np.linspace(x0[1,0]-tmp_eig, x0[1,0]+tmp_eig, 600)
	xxx, yyy = np.meshgrid(xx, yy)
	result=f(xxx, yyy)
	ax.contour(xxx, yyy, result, levels=[0])

	#OTHER OPTION, but you run into this visualization issue: https://github.com/sympy/sympy/issues/20056
	# tmp=sympy.plot_implicit(expression,show=True,points=300, adaptive=False, depth = 2)
	# move_sympyplot_to_axes(tmp, ax)
	# pts = tmp.get_points()
	# plt.plot(pts[0], pts[1])
	

# E representation --> {x s.t. (x-x0)'E(x-x0) <= 1}. Here, E is a psd matrix
# B representation --> {x s.t. x=B*p_bar + x0, ||p_bar||<=1} \equiv {x s.t. ||inv(B)(x-x0)||<=1} \equiv {x s.t. (x-x0)'*inv(B)'*inv(B)*(x-x0)<=1}. 
# B is \in R^nxn (although Boyd's book says we can assume B is psd (and therefore also symmetric) without loss of generality, see section 8.4.1
# More info about the B representation: https://ieeexplore.ieee.org/abstract/document/7839930

#It returns the ellipsoid in B representation
#It operates on numpy stuff 
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

#Operates on torch stuff
def squared_norm_of_each_row(D):
	return (D**2)@torch.ones((D.shape[1],1))

#Operates on torch stuff
def scaleEllipsoidB(B,A,b,x0):

	# print("\n\n")
	# # #====================First way==========================
	# # ========================================================
	# minimum_so_far=torch.Tensor([float("inf")])
	# for i in range(torch.numel(b)):
	# 	# print(A[i,:])
	# 	a_i=makeColumnVector(A[i,:])
		
	# 	tmp=(b[i,0]-a_i.mT@x0)**2/(a_i.mT@B@B.T@a_i);
	# 	# print(f"numerator={(b[i,0]-a_i.mT@x0)**2}, denominator={a_i.mT@B@B.T@a_i}, result={tmp}")
	# 	# print(f"tmp is {tmp}")
	# 	# print(f"tmp[0,0] is {tmp[0]}")
	# 	minimum_so_far=torch.minimum(minimum_so_far, tmp[0,0])

	# # print(f"-------> minimum so far={minimum_so_far}")

	# result = B*torch.sqrt(minimum_so_far);
	# print(f"First way: \n {result}")

	# # #===================Second way==========================
	# # ========================================================
	# c=squared_norm_of_each_row(A@B)
	# e=torch.min(((b-A@x0)**2)/c)
	# result=B*torch.sqrt(e)
	# print(f"Second way: \n {result}")
	
	# #===================Third way==========================
	# ========================================================
	sqrt_c=torch.sqrt(squared_norm_of_each_row(A@B))
	sqrt_e=torch.min(torch.abs(b-A@x0)/sqrt_c)#Note that if x0 is inside the ellipsoid, then I don't need the abs(), since Ax0<=b --> b-Ax0>=0
	result=B*sqrt_e
	# print(f"Third way: \n {result}")


	return result;

class MyLayer(torch.nn.Module):
	def __init__(self, A_np, b_np, num_steps):
		super().__init__()

		self.dim=A_np.shape[1]
		self.num_steps=num_steps;
		B_np, x0_np = largestEllipsoidBInPolytope(A_np,b_np)
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
			B_last=scaleEllipsoidB(self.B,self.A,self.b,x0_last) #Note that here, self.B (instead of B_last) is preferred for numerical stability 
			self.all_B.append(B_last)
			self.all_x0.append(x0_last)

		return x0_last

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

# x0=torch.tensor(np.array([[0.5],[0.2]]));
# B=scaleEllipsoidB(torch.tensor(B),torch.tensor(A),torch.tensor(b),x0)

# print(f"Using x0={x0} and scaling gives B={B}")

# print(B)
# print(x0)

# a=np.array([[4], [8], [9], [7], [2], [8],[5], [1], [0.5],]);
num_steps=2;
my_layer=MyLayer(A,b,num_steps)

all_optimal_points=torch.tensor(np.array([[],[]],dtype=np.float32))

fig, ax = plt.subplots()

#for j in range(10000):
for theta in np.arange(0,2*math.pi, 0.01): #[0.93]: #
	x=torch.Tensor(np.array([[math.cos(theta)],[math.sin(theta)],[3000]]));
	# x=torch.Tensor(my_layer.input_numel, 1).uniform_(-1, 1)
	# x=torch.Tensor(np.array([[0.5],[4],[3]]));
	x= x.repeat(num_steps, 1)
	# print(f"x={x}")
	# print(f"x.mT={x.mT}")
	optimal_point=my_layer.forward(x)
	#print(f"optimal_point.mT={optimal_point.mT}")
	all_optimal_points=torch.cat((all_optimal_points,optimal_point),1)
	print(f"Theta={theta}")
	# my_layer.plotAllSteps(ax)


print(f"all_optimal_points={all_optimal_points}");
all_optimal_points_np=all_optimal_points.numpy();


# plot

ax.scatter(all_optimal_points_np[0,:], all_optimal_points_np[1,:])

plot2DPolyhedron(A,b,ax)

plot2DEllipsoidB(B,x0,ax)

plt.show()

# %%
