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
from tqdm import tqdm

def printInBoldBlue(data_string):
    print(Style.BRIGHT+Fore.BLUE+data_string+Style.RESET_ALL)
def printInBoldRed(data_string):
    print(Style.BRIGHT+Fore.RED+data_string+Style.RESET_ALL)
def printInBoldGreen(data_string):
    print(Style.BRIGHT+Fore.GREEN+data_string+Style.RESET_ALL)


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

#Everything in numpy
class LinearConstraint():
	def __init__(self, Aineq, bineq, Aeq, beq):
		self.Aineq = Aineq
		self.bineq = bineq
		self.Aeq = Aeq
		self.beq = beq

		# x0 = cp.Variable((n,1))

		################################# CHECKS
		self.has_eq_constraints=((Aeq is not None) and (beq is not None));
		self.has_ineq_constraints=((Aineq is not None) and (bineq is not None));

		if(self.has_ineq_constraints):
			assert Aineq.ndim == 2, f"Aineq.shape={Aineq.shape}"
			assert bineq.ndim == 2, f"bineq.shape={bineq.shape}"
			assert bineq.shape[1] ==1
			assert Aineq.shape[0] == bineq.shape[0]
			self.dim_ambient_space=Aineq.shape[1]

		if(self.has_eq_constraints):
			assert Aeq.ndim == 2, f"Aeq.shape={Aeq.shape}"
			assert beq.ndim == 2, f"beq.shape={beq.shape}"
			assert beq.shape[1] ==1
			assert Aeq.shape[0] == beq.shape[0]
			self.dim_ambient_space=Aeq.shape[1]

		if(self.has_eq_constraints and self.has_ineq_constraints):
			assert Aineq.shape[1] == Aeq.shape[1]

		if(self.has_eq_constraints==False and self.has_ineq_constraints==False):
			raise Exception("There are no constraints!")
		#################################

		###################### FOR PROJECTION
		#Section 8.1.1 of https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
		self.x_projected = cp.Variable((self.dimAmbSpace(),1))         #projected point
		self.x_to_be_projected = cp.Parameter((self.dimAmbSpace(),1))  #original point
		constraints=self.getCvxpyConstraints(self.x_projected)
		objective = cp.Minimize(cp.sum_squares(self.x_projected - self.x_to_be_projected))
		self.prob_projection = cp.Problem(objective, constraints)


	def hasIneqConstraints(self):
		return self.has_ineq_constraints

	def hasEqConstraints(self):
		return self.has_eq_constraints

	def dimAmbSpace(self):
		return self.dim_ambient_space

	def getCvxpyConstraints(self, variable):
		constraints=[]
		assert variable.shape[0]==self.dim_ambient_space 
		assert variable.shape[1]==1 
		if(self.hasIneqConstraints()):
			constraints.append(self.Aineq@variable<=self.bineq)
		if(self.hasEqConstraints()):
			constraints.append(self.Aeq@variable==self.beq)   
		return constraints 	

	def project(self, x_to_be_projected):
		assert x_to_be_projected.shape==self.x_to_be_projected.shape

		self.x_to_be_projected.value=x_to_be_projected;
		obj_value = self.prob_projection.solve(verbose=False);
		if(self.prob_projection.status != 'optimal' and self.prob_projection.status != 'optimal_inaccurate'):
			raise Exception(f"Value is not optimal, prob_status={self.prob_projection.status}")

		return self.x_projected.value, obj_value	

	def getViolation(self, x_to_be_projected):

		if(x_to_be_projected.ndim==1):
			#convert to a column vector
			x_to_be_projected=np.expand_dims(x_to_be_projected, axis=1)

		# print(f"projecting {x_to_be_projected}")

		_, violation = self.project(x_to_be_projected)

		assert violation>=0  #violation is nonnegative by definition

		return violation;

		# return x_projected

		# x = cp.Variable((self.dimAmbSpace(),1))
		# assert x0.shape==x.shape
		# constraints=self.getCvxpyConstraints(x);
		# print(f"All the constraints: {constraints}")

		# print(f"First constraint: {constraints[0]}")
		# print(f"Second constraint: {constraints[1]}")

		# print(f"bineq-Aineq*x={self.bineq-self.Aineq@x0}")
		# print(f"self.beq-self.Aeq@x0={self.beq-self.Aeq@x0}")

		# objective = cp.Minimize(0.0)
		# prob = cp.Problem(objective, constraints)
		# result = prob.solve(verbose=False);

		# print(prob.constraints)

		# # x.value=x0;
		# print(f"violation={(prob.constraints).violation()}")
		# exit()

		# constraints.


	def process(self):

		z = cp.Variable((self.dimAmbSpace(),1))

		TOL=1e-7;

		################################################
		#Make sure that the LP has a feasible solution
		objective = cp.Minimize(0.0)
		constraints=self.getCvxpyConstraints(z)
		prob = cp.Problem(objective, constraints)
		result = prob.solve(verbose=False);
		if(prob.status != 'optimal'):
			raise Exception("The feasible set is empty")
		################################################

		if(self.hasIneqConstraints()):
			A=self.Aineq;
			b=self.bineq;
			if(self.hasEqConstraints()): 
				#Add the equality constraints as inequality constraints
				A=np.concatenate((A,self.Aeq,-self.Aeq), axis=0);
				b=np.concatenate((b,self.beq,-self.beq), axis=0);				
		else:
			#Add the equality constraints as inequality constraints
			A=np.concatenate((self.Aeq,-self.Aeq), axis=0);
			b=np.concatenate((self.beq,-self.beq), axis=0);

		#At this point, the feasible region is represented by Ax<=b		

		# printInBoldBlue("-- point1")
		# print(f"A=\n{A}\n b=\n{b}")

		printInBoldGreen(f"A is {A.shape} and b is {b.shape}")

		#Remove redundant constraints
		################################################
		#Eq 1.5 of https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/167108/1/thesisFinal_MaySzedlak.pdf
		#See also https://mathoverflow.net/a/69667
		printInBoldBlue("Removing redundant constraints...")
		indexes_const_removed=[]
		reversed_indexes=list(reversed(range(A.shape[0])));
		for i in tqdm(reversed_indexes):
			# print(i)
			all_rows_but_i=[x for x in range(A.shape[0]) if x != i]
			objective = cp.Maximize(A[i,:]@z)
			constraints=[A[all_rows_but_i,:]@z<=b[all_rows_but_i,:],   A[i,:]@z<=(b[i,0]+1)]
			prob = cp.Problem(objective, constraints)
			result = prob.solve(verbose=False);
			if(prob.status != 'optimal'):
				raise Exception("Value is not optimal")

			if ((objective.value-b[i,0])<=TOL):
				# print(f"Deleting constraint {i}")
				indexes_const_removed.append(i)
				A = np.delete(A, (i), axis=0)
				b = np.delete(b, (i), axis=0)

		# printInBoldBlue(f"Removed constraints {indexes_const_removed}")
		printInBoldBlue(f"Removed {len(indexes_const_removed)} constraints ")
		printInBoldGreen(f"A is {A.shape} and b is {b.shape}")

		# print(f"A=\n{A}\n b=\n{b}")

		#Find equality set
		################################################
		# Section 5.2 of https://www.researchgate.net/publication/268373838_Polyhedral_Tools_for_Control
		# See also Definition 2.16 of https://sites.math.washington.edu/~thomas/teaching/m583_s2008_web/main.pdf

		E=[] #contains the indexes of the constraints in the equality set

		# print(f"A=\n{A}")
		# print(f"b=\n{b}")

		printInBoldBlue("Finding Affine Hull and projecting...")

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


		printInBoldGreen(f"E={E}")

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
			A_I=np.zeros((1,A.shape[1]));
			b_I=np.zeros((1,1));	

		#At this point, A_E, b_E, A_I, and b_I have at least one row

		#Project into the nullspace of A_E
		################################################
		NA_E=scipy.linalg.null_space(A_E);
		n=NA_E.shape[1] #dimension of the subspace
		p0=np.linalg.pinv(A_E)@b_E
		A_p=A_I@NA_E;
		b_p=b_I-A_I@p0
				

		print(f"A_p=\n{A_p}")
		print(f"b_p=\n{b_p}")

		assert A_p.ndim == 2, f"Aineq.shape={A_p.shape}"
		assert b_p.ndim == 2, f"bineq.shape={b_p.shape}"
		assert b_p.shape[1] ==1
		assert A_p.shape[0] == b_p.shape[0]

		# print(f"A=\n{A}")
		# print(f"b=\n{b}")

		printInBoldGreen(f"A_p is {A_p.shape} and b_p is {b_p.shape}")

		#Check whether or not the polyhedron is bounded
		###############################################
		# See https://github.com/TobiaMarcucci/pympc/blob/master/pympc/geometry/polyhedron.py#L529
		# and https://math.stackexchange.com/a/3593310/564801
		bounded=True
		print(f"A_p.shape={A_p.shape}")
		NA_p=scipy.linalg.null_space(A_p);
		if(NA_p.size!=0): #if the null(A_p)!={0}
			bounded=False
		else: #if the null(A_p)=={0} (i.e., if A_p is invertible) 
			n0=A_p.shape[0];
			n1=A_p.shape[1];
			y = cp.Variable((n0,1))
			objective = cp.Minimize(cp.norm1(y))
			constraints=[A_p.T@y==np.zeros((n1,1)),  y>=np.ones((n0,1))]
			prob = cp.Problem(objective, constraints)
			result = prob.solve(verbose=False);
			if(prob.status == 'infeasible'):
				bounded=False  

		if(bounded):
			printInBoldBlue("Bounded feasible set")
		else:
			printInBoldGreen("Unbounded feasible set")


		if(bounded):
			max_radius=None
		else:
			max_radius=1.0; #to prevent unbounded result

		B, z0 = largestBallInPolytope(A_p,b_p, max_radius) 
		#B, z0 = largestEllipsoidBInPolytope(A,b) #This is very slow in high dimensions

		return A_p, b_p, p0, NA_E, B, z0



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

	printInBoldRed(f"Found {len(V_list)} vertices and {len(R_list)} rays")

	V = np.asarray(V_list)
	R = np.asarray(R_list)

	# by convention of cddlib, those rays assciated with R_lin_idx
	# are not constrained to non-negative coefficients
	if len(R) > 0:
		R = np.concatenate([R, -R[R_lin_idx, :]], axis=0)

	V=V.T; 
	R=R.T;


	if(R.size==0):
		R=np.array([[]])#Simply add a dimension, so that both V and R are 2D matrices

	if(V.size==0):
		V=np.array([[]])#Simply add a dimension, so that both V and R are 2D matrices

	#Each column of V is a vertex
	#Each column of R is a ray

	return V, R


def plot3DPolytopeHRepresentation(A,b, limits, ax):
	points, R=H_to_V(A,b)
	plot3DPolytopeVRepresentation(points, limits, ax)

def plot3DPolytopeVRepresentation(V, limits, ax):
	points=V.T

	## https://stackoverflow.com/a/71544694/6057617

	# to get the convex hull with cdd, one has to prepend a column of ones
	vertices = np.hstack((np.ones((8,1)), points))

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

	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")

	ax.set_xlim3d(limits[0],limits[1])
	ax.set_ylim3d(limits[2],limits[3])
	ax.set_zlim3d(limits[4],limits[5])



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