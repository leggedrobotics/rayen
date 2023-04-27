import cdd
from scipy.spatial import ConvexHull
import cvxpy as cp
import numpy as np
import torch
import sympy
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
import torch.nn as nn
import typing
import scipy
import math

def printInBoldBlue(data_string):
	print(Style.BRIGHT+Fore.BLUE+data_string+Style.RESET_ALL)
def printInBoldRed(data_string):
	print(Style.BRIGHT+Fore.RED+data_string+Style.RESET_ALL)
def printInBoldGreen(data_string):
	print(Style.BRIGHT+Fore.GREEN+data_string+Style.RESET_ALL)
def printInBoldWhite(data_string):
	print(Style.BRIGHT+Fore.WHITE+data_string+Style.RESET_ALL)


def getAllPqrFromQcs(qcs):
		all_P=[]
		all_q=[]
		all_r=[]
		for qc in qcs:
			all_P.append(qc.P)
			all_q.append(qc.q)
			all_r.append(qc.r)	
		return all_P, all_q, all_r

def getAllMscdFromQcs(socs):
		all_M=[]
		all_s=[]
		all_c=[]
		all_d=[]
		for soc in socs:
			all_M.append(soc.M)
			all_s.append(soc.s)
			all_c.append(soc.c)
			all_d.append(soc.d)

		return all_M, all_s, all_c, all_d

#Ellisoid is represented by {x | x'*E*x <=1}
def plotEllipsoid(E, x0, ax):
	#Partly taken from https://github.com/CircusMonkey/covariance-ellipsoid/blob/master/ellipsoid.py
	"""
	Return the 3d points representing the covariance matrix
	cov centred at mu and scaled by the factor nstd.
	Plot on your favourite 3d axis. 
	Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
	Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)
	"""

	assert E.shape==(3,3)

	B=np.linalg.inv(scipy.linalg.sqrtm(E))

	#Ellisoid is now represented by { Bp+x0 | ||p|| <=1}
	# Find and sort eigenvalues 
	eigvals, eigvecs = np.linalg.eigh(B)
	idx = np.sum(B,axis=0).argsort()
	eigvals_temp = eigvals[idx]
	idx = eigvals_temp.argsort()
	eigvals = eigvals[idx]
	eigvecs = eigvecs[:,idx]

	# Set of all spherical angles to draw our ellipsoid
	n_points = 100
	theta = np.linspace(0, 2*np.pi, n_points)
	phi = np.linspace(0, np.pi, n_points)

	# Width, height and depth of ellipsoid
	rx, ry, rz = np.sqrt(eigvals)

	# Get the xyz points for plotting
	# Cartesian coordinates that correspond to the spherical angles:
	X = rx * np.outer(np.cos(theta), np.sin(phi))
	Y = ry * np.outer(np.sin(theta), np.sin(phi))
	Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

	# Rotate ellipsoid for off axis alignment
	old_shape = X.shape
	# Flatten to vectorise rotation
	X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
	X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
	X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
	# Add in offsets for the center
	X = X + x0[0]
	Y = Y + x0[1]
	Z = Z + x0[2]

	ax.plot_wireframe(X,Y,Z, color='r', alpha=0.1)

#It operates on numpy stuff 
#polytope defined as Ax<=b
def largestBallInPolytope(A,b, max_radius=None):

	if len(b.shape) == 1:
		b = np.expand_dims(b, axis=1) #Make b a column vector

	n=A.shape[1];
	r = cp.Variable()#A scalar
	x0 = cp.Variable((n,1))
	constraints=[]

	#https://programtalk.com/vs2/python/2718/cvxpy/examples/chebyshev.py/
	#See also https://dkenefake.github.io/blog/ChebBall for when there are equality constraints
	for i in range(A.shape[0]):
		a_i=A[i,:].T
		constraints.append(r*cp.norm(a_i)+a_i.T@x0<=b[i,0])

	if(max_radius is not None):
		constraints.append(r<=max_radius)

	objective = cp.Minimize(-r) #This is just a linear program
	prob = cp.Problem(objective, constraints)
	print("Calling solve...")
	result = prob.solve(verbose=False);
	print("Solved!")
	if(prob.status != 'optimal'):
		raise Exception("Value is not optimal")

	B=r*np.eye(n)

	printInBoldGreen(f"Found ball of radius r={r.value}")

	return B.value, x0.value


def isZero(A):
	return (not np.any(A))

def checkMatrixisNotZero(A):
	assert (not isZero(A))

def checkMatrixisSymmetric(A):
	assert A.shape[0]==A.shape[1]
	assert np.allclose(A, A.T)

def checkMatrixisPsd(A):
	checkMatrixisSymmetric(A)
	eigenvalues=np.linalg.eigvals(A);
	assert np.all(eigenvalues >= 0.0), f"Matrix is not PSD, min eigenvalue is {np.amin(eigenvalues)}"

def checkMatrixisPd(A):
	checkMatrixisSymmetric(A)
	eigenvalues=np.linalg.eigvals(A);
	assert np.all(eigenvalues > 0.0), f"Matrix is not PD, min eigenvalue is {np.amin(eigenvalues)}"

def isMatrixSingular(A):
	return (np.linalg.matrix_rank(A) < self.E.shape[0])

# Removes redundant equations from Ax=b
# Note that 
#Ax=b                     is equivalent to
#[A b][x;-1]=0            Performing now the QR decomposition we have:
#QR[x;-1]=0               As Q^T=Q^-1, we have that:
#R[x;-1]=0                Noting now that R=[Rnz;0], we have that:
#[Rnz;0][x;-1]=0          Here, 0 is a matrix/vector of zeros. Getting rid of the zero part
#[Rnz][x;-1]=0            Denoting now Rnz:=[Aresult bresult]
# Aresult x = bresult     This is the equivalent system to Ax=b
def removeRedundantEquationsFromEqualitySystem(A, b):
	A_b=np.concatenate((A, b), axis=1)

	# (_, rref1) = scipy.linalg.qr(A_b) 
	# rref2=sympy.Matrix(A_b).rref();

	#See section titled "Alternatively" of https://stackoverflow.com/a/39621887
	#See also http://www.ryanhmckenna.com/2021/03/removing-redundant-constraints-from.html
	(_, R) = np.linalg.qr(A_b) 

	rows_that_are_zero=[]
	for i in range(R.shape[0]):
		if(np.linalg.norm(R[i,:])<1e-7):
			rows_that_are_zero.append(i)


	Rnz = np.delete(R, rows_that_are_zero, axis=0)

	A_result=Rnz[:,:-1].reshape((-1, A.shape[1]))
	b_result=Rnz[:,-1].reshape((-1, 1))

	return A_result, b_result

#Everything in numpy
#Ellipsoid is defined as {x | (x-c)'E(x-c)<=1}
#Where E is a positive semidefinite matrix
class Ellipsoid():
	def __init__(self, E, c):

		checkMatrixisPsd(E);
		self.E=E;
		self.c=c;

	def convertToQuadraticConstraint(self):
		P=2*self.E;
		q=(-2*self.E@self.c)
		r=self.c.T@self.E@self.c-1
		return convexQuadraticConstraint(P,q,r)

#Pytorch
def quadExpression(y, P, q, r):

	P = P.to(y.device)
	q = q.to(y.device)
	r = r.to(y.device)

	# print(f"P.shape={P.shape}")
	# print(f"q.shape={q.shape}")
	# print(f"r.shape={r.shape}")
	# print(f"y.shape={y.shape}")

	if (q.ndim==2):
		qT=q.T
	else: #q is a batch
		assert q.ndim==3 
		qT=torch.transpose(q,1,2)

	result=0.5*torch.transpose(y,1,2)@P@y + qT@y + r;
	assert result.shape==(y.shape[0],1,1)
	return result

#https://stackoverflow.com/a/3844832
def all_equal(iterator):
	iterator = iter(iterator)
	try:
		first = next(iterator)
	except StopIteration:
		return True
	return all(first == x for x in iterator)



def loadpickle(name_file):
	import pickle
	with open(name_file, "rb") as input_file:
		result = pickle.load(input_file)
	return result	

def savepickle(variable, name_file):
	import pickle
	with open(name_file, 'wb') as output_file:
		pickle.dump(variable, output_file, pickle.HIGHEST_PROTOCOL)	

#This function has been taken (and modified) from https://github.com/DLR-RM/stable-baselines3/blob/201fbffa8c40a628ecb2b30fd0973f3b171e6c4c/stable_baselines3/common/torch_layers.py#L96
def create_mlp(
	input_dim: int,
	output_dim: int,
	net_arch: typing.List[int],
	activation_fn: typing.Type[nn.Module] = nn.ReLU,
	squash_output: bool = False):
	"""
	Create a multi layer perceptron (MLP), which is
	a collection of fully-connected layers each followed by an activation function.
	:param input_dim: Dimension of the input vector
	:param output_dim:
	:param net_arch: Architecture of the neural net
		It represents the number of units per layer.
		The length of this list is the number of layers.
	:param activation_fn: The activation function
		to use after each layer.
	:param squash_output: Whether to squash the output using a Tanh
		activation function
	:return:
	"""

	if len(net_arch) > 0:
		modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
	else:
		modules = []

	for idx in range(len(net_arch) - 1):
		modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
		modules.append(activation_fn())

	if output_dim > 0:
		last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
		modules.append(nn.Linear(last_layer_dim, output_dim))
	if squash_output:
		modules.append(nn.Tanh())
	return nn.Sequential(*modules)

#https://stackoverflow.com/questions/65154622/sample-uniformly-at-random-from-a-simplex-in-python
def runif_in_simplex(n):
  ''' Return uniformly random vector in the n-simplex '''

  k = np.random.exponential(scale=1.0, size=n)
  return k / sum(k)

#This function is taken from https://github.com/tfrerix/constrained-nets
def H_to_V(A, b):
	"""
	Converts a polyhedron in H-representation to
	one in V-representation using pycddlib.
	"""
	# define cdd problem and convert representation
	if len(b.shape) == 1:
		b = np.expand_dims(b, axis=1)
	mat_np = np.concatenate([b, -A], axis=1)
	if mat_np.dtype in [np.int32, np.int64]:
		nt = 'fraction'
	else:
		nt = 'float'
	mat_list = mat_np.tolist()

	mat_cdd = cdd.Matrix(mat_list, number_type=nt)
	mat_cdd.rep_type = cdd.RepType.INEQUALITY
	poly = cdd.Polyhedron(mat_cdd)
	gen = poly.get_generators()

	# convert the cddlib output data structure to numpy
	V_list = []
	R_list = []
	lin_set = gen.lin_set
	V_lin_idx = []
	R_lin_idx = []
	for i in range(gen.row_size):
		g = gen[i]
		g_type = g[0]
		g_vec = g[1:]
		if i in lin_set:
			is_linear = True
		else:
			is_linear = False
		if g_type == 1:
			V_list.append(g_vec)
			if is_linear:
				V_lin_idx.append(len(V_list) - 1)
		elif g_type == 0:
			R_list.append(g_vec)
			if is_linear:
				R_lin_idx.append(len(R_list) - 1)
		else:
			raise ValueError('Generator data structure is not valid.')

	V = np.asarray(V_list)
	R = np.asarray(R_list)

	# by convention of cddlib, those rays associated with R_lin_idx
	# are not constrained to non-negative coefficients
	if len(R) > 0:
		R = np.concatenate([R, -R[R_lin_idx, :]], axis=0)

	V=V.T; 
	R=R.T;

	if(R.size==0):
		R=np.array([[]])#Simply add a dimension, so that both V and R are 2D matrices

	if(V.size==0):
		V=np.array([[]])#Simply add a dimension, so that both V and R are 2D matrices

	printInBoldRed(f"Found {V.shape[1]} vertices and {R.shape[1]} rays")

	#Each column of V is a vertex
	#Each column of R is a ray

	return V, R


def plot3DPolytopeHRepresentation(A,b, limits, ax):
	points, R=H_to_V(A,b)
	if(R.shape[1]>0):
		printInBoldRed("Plotting 3D unbounded polyhedron not implemented yet")
		return
	plotConvexHullOf3DPoints(points, limits, ax)

def plotConvexHullOf3DPoints(V, limits, ax):
	points=V.T

	## https://stackoverflow.com/a/71544694/6057617

	# to get the convex hull with cdd, one has to prepend a column of ones
	vertices = np.hstack((np.ones((points.shape[0],1)), points))

	# do the polyhedron
	mat = cdd.Matrix(vertices, linear=False, number_type="fraction") 
	mat.rep_type = cdd.RepType.GENERATOR
	poly = cdd.Polyhedron(mat)

	# get the adjacent vertices of each vertex
	adjacencies = [list(x) for x in poly.get_input_adjacency()]

	# store the edges in a matrix (giving the indices of the points)
	edges = [None]*(8-1)
	for i,indices in enumerate(adjacencies[:-1]):
		indices = list(filter(lambda x: x>i, indices))
		l = len(indices)
		col1 = np.full((l, 1), i)
		indices = np.reshape(indices, (l, 1))
		edges[i] = np.hstack((col1, indices))
	Edges = np.vstack(tuple(edges))

	# plot
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection="3d")

	start = points[Edges[:,0]]
	end = points[Edges[:,1]]

	for i in range(12):
		ax.plot(
			[start[i,0], end[i,0]], 
			[start[i,1], end[i,1]], 
			[start[i,2], end[i,2]],
			"blue"
		)

	# ax.set_xlabel("x")
	# ax.set_ylabel("y")
	# ax.set_zlabel("z")

	# ax.set_xlim3d(limits[0],limits[1])
	# ax.set_ylim3d(limits[2],limits[3])
	# ax.set_zlim3d(limits[4],limits[5])



from mpl_toolkits.mplot3d import axes3d

#Taken from here: https://stackoverflow.com/a/4687582/6057617
def plot_implicit(fn, limits):
	''' create a plot of an implicit function
	fn  ...implicit function (plot where fn==0)
	bbox ..the x,y,and z limits of plotted interval'''
	# xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	A = np.linspace(limits[0],limits[1], 100) # resolution of the contour
	B = np.linspace(limits[0],limits[1], 15) # number of slices
	A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

	for z in B: # plot contours in the XY plane
		X,Y = A1,A2
		Z = fn(X,Y,z)
		cset = ax.contour(X, Y, Z+z, [z], zdir='z')
		# [z] defines the only level to plot for this contour for this value of z

	for y in B: # plot contours in the XZ plane
		X,Z = A1,A2
		Y = fn(X,y,Z)
		cset = ax.contour(X, Y+y, Z, [y], zdir='y')

	for x in B: # plot contours in the YZ plane
		Y,Z = A1,A2
		X = fn(x,Y,Z)
		cset = ax.contour(X+x, Y, Z, [x], zdir='x')

	# must set plot limits because the contour will likely extend
	# way beyond the displayed level.  Otherwise matplotlib extends the plot limits
	# to encompass all values in the contour.
	ax.set_xlim3d(limits[0],limits[1])
	ax.set_ylim3d(limits[2],limits[3])
	ax.set_zlim3d(limits[4],limits[5])

	plt.show()


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

def uniformSampleInUnitSphere(dim):
	#Method 19 of http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

	u = np.random.normal(loc=0.0, scale=1.0, size=(dim,1))
	u_normalized= u / np.linalg.norm(u)

	return u_normalized

# https://stackoverflow.com/a/69427715/6057617
# def move_sympyplot_to_axes(p, ax):
#     backend = p.backend(p)
#     backend.ax = ax
#     backend._process_series(backend.parent._series, ax, backend.parent)
#     backend.ax.spines['right'].set_color('none')
#     backend.ax.spines['top'].set_color('none')
#     backend.ax.spines['bottom'].set_position('zero')
#     plt.close(backend.fig)

def plot2DPolyhedron(A, b, ax):
	##FIRST WAY
	# npoints=300
	# d = np.linspace(-2,16,npoints)
	# x1,x2 = np.meshgrid(d,d)

	# tmp=1;
	# for i in range(A.shape[0]):
	# 	tmp=tmp & (A[i,0]*x1 + A[i,1]*x2 <=b[i,0]);

	# plt.imshow( tmp.astype(int), extent=(x1.min(),x1.max(),x2.min(),x2.max()),origin="lower", cmap="Greys", alpha = 0.3);

	# #second way, right now it only works if there are no rays
	# vertices, rays = getVertexesRaysFromAb(A, b)
	# #See example from https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
	# coord = vertices.T
	# hull = ConvexHull(coord)
	# for simplex in hull.simplices:
	# 	ax.plot(coord[simplex, 0], coord[simplex, 1], 'r-')

	#third way
	npoints=300
	d = np.linspace(-2,16,npoints)
	x1,x2 = np.meshgrid(d,d)

	tmp=1;
	for i in range(A.shape[0]):
		tmp=tmp & (A[i,0]*x1 + A[i,1]*x2 <=b[i,0]);

	plt.imshow( tmp.astype(int), extent=(x1.min(),x1.max(),x2.min(),x2.max()),origin="lower", cmap="Greys", alpha = 0.3);


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

	if len(b.shape) == 1:
		b = np.expand_dims(b, axis=1) #Make b a column vector

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
	print("Calling solve...")
	result = prob.solve(verbose=True);
	print("Solved!")

	return B.value, x0.value

def makeColumnVector(a):
	return a[:,None]

#Operates on torch stuff
def squared_norm_of_each_row(D):
	# print(f"D**2.shape={(D**2).shape}")
	result=torch.sum(D**2, dim=2, keepdim=True)
	# print(f"result.shape={result.shape}")

	return result  #@torch.ones((D.shape[1],1),device=D.device)

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
	
	sqrt_c=torch.sqrt(squared_norm_of_each_row(A@B)) #This could be computed in the constructor (and not in each forward call)

	# print(f"sqrt_c={sqrt_c}")


	Ax0=A@x0;
	b_minus_Ax0=torch.sub(torch.unsqueeze(b,dim=0),Ax0)
	abs_b_minus_Ax0=torch.abs(b_minus_Ax0) #Note that if x0 is inside the ellipsoid, then I don't need the abs(), since Ax0<=b --> b-Ax0>=0
	abs_b_minus_Ax0_divided_by_sqrt_c=torch.div(abs_b_minus_Ax0,sqrt_c)
	tmp=torch.min(abs_b_minus_Ax0_divided_by_sqrt_c,dim=1,keepdim=True)
	sqrt_e=tmp.values
	result=B*sqrt_e
	# print(f"Third way: \n {result}")

	return result;


	# sqrt

	# print(f"sqrt_e={sqrt_e}")


	# print(f"sqrt_e.shape={sqrt_e.shape}")
	# print(f"B.shape={B.shape}")


	# print("-------------\n")
	# print(f"x0.shape={x0.shape}")
	
	# print(f"B.shape={B.shape}")
	# print(f"A.shape={A.shape}")
	# print(f"A@B.shape={(A@B).shape}")
	# print(f"b.shape={b.shape}")


	# print("==============================")
	# print(f"A={A}")
	# print(f"B={B}")
	# print(f"b={b}")
	# print(f"x0={x0}\n")