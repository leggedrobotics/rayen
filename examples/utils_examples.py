# --------------------------------------------------------------------------
# Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
# See LICENSE file for the license information
# -------------------------------------------------------------------------- 

import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
import cdd
import fixpath
from rayen import utils

########################################################
########################################################

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


def plot3DPolytopeHRepresentation(A,b, limits, ax):
	points, R=utils.H_to_V(A,b)
	if(R.shape[1]>0):
		utils.printInBoldRed("Plotting 3D unbounded polyhedron not implemented yet")
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
	# from scipy.spatial import ConvexHull 
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



	