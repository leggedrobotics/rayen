# --------------------------------------------------------------------------
# Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
# See LICENSE file for the license information
# -------------------------------------------------------------------------- 
import cdd
import numpy as np
import torch
from colorama import Fore, Back, Style
import torch.nn as nn

def printInBoldBlue(data_string):
	print(Style.BRIGHT+Fore.BLUE+data_string+Style.RESET_ALL)
def printInBoldRed(data_string):
	print(Style.BRIGHT+Fore.RED+data_string+Style.RESET_ALL)
def printInBoldGreen(data_string):
	print(Style.BRIGHT+Fore.GREEN+data_string+Style.RESET_ALL)
def printInBoldWhite(data_string):
	print(Style.BRIGHT+Fore.WHITE+data_string+Style.RESET_ALL)


def verify(condition, message="Condition not satisfied"):
	if(condition==False):
		raise RuntimeError(message)

def getAllPqrFromQcs(qcs):
		all_P=[]
		all_q=[]
		all_r=[]
		for qc in qcs:
			all_P.append(qc.P)
			all_q.append(qc.q)
			all_r.append(qc.r)	
		return all_P, all_q, all_r

def getAllMscdFromSocs(socs):
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


class CudaTimer():
	def __init__(self):
			pass
	def start(self):
			#See https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
			self.start_event = torch.cuda.Event(enable_timing=True)
			self.end_event = torch.cuda.Event(enable_timing=True)
			self.start_event.record()

	def endAndGetTimeSeconds(self):
			self.end_event.record()
			torch.cuda.synchronize()
			return  (1e-3)*self.start_event.elapsed_time(self.end_event) #Note that elapsed_time returns the time in milliseconds


########################################################
########################################################

# Inspired by https://github.com/rfeinman/Torch-ARPACK/blob/master/arpack/power_iteration.py
# See also https://ergodic.ugr.es/cphys/LECCIONES/FORTRAN/power_method.pdf
# Note that powerIteration find the eigenvalue with largest absolute value (i.e., the dominant eigenvalue)
# v is the initial guess of the eigenvector associated with the dominant eigenvalue
# A should have shape [size_batch, n, n]
# v should have shape [n, 1]
# Everything in Pytorch
def powerIteration(A, v, tol=1e-5, max_iter=100000, eps=1e-12, check_freq=2):
		n_iter = 0
		v = torch.nn.functional.normalize(v)
		while n_iter < max_iter:
				n_iter += 1
				u = torch.nn.functional.normalize(A@v, dim=1)

				if n_iter>1 and ((n_iter % check_freq) == 0):
						distance=torch.mean(torch.abs(1 - torch.abs(  torch.transpose(v,1,2)@u ))) #Note: one disadvantage of this is that it will keep iterating until ALL the elements of the batch have converged
						if distance<tol:
								v = u
								break
				v = u
		else:
				print(f"distance={distance}")
				print('Power iteration did not converge')

		lamb =  torch.transpose(v,1,2)@A@v #torch.dot(v, torch.mv(A, v))

		return lamb

#See https://math.stackexchange.com/a/835568 (note that there are some signs wrong in that answer, but otherwise is good)
# v is the initial guess for the power iteration
# A should have shape [size_batch, n, n]
# v should have shape [n, 1]
# Everything in Pytorch
def findLargestEigenvalueUsingPowerIteration(A, v):
		lamb = powerIteration(A, v)
		condition=(lamb.flatten()<0)
		if (torch.all(condition == False)):
				return lamb
		lamb[condition,:,:]+=powerIteration(A[condition,:,:]-lamb[condition,:,:]*torch.eye(A.shape[1]), v)
		return lamb


# Other links: 
# --> https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/spectral_norm.py#L80


def isZero(A):
	return (not np.any(A))

def checkMatrixisNotZero(A):
	verify(not isZero(A))

def checkMatrixisSymmetric(A):
	verify(A.shape[0]==A.shape[1])
	verify(np.allclose(A, A.T))

def checkMatrixisPsd(A, tol=0.0):
	checkMatrixisSymmetric(A)
	eigenvalues=np.linalg.eigvals(A);
	verify(np.all(eigenvalues >= -tol), f"Matrix is not PSD, min eigenvalue is {np.amin(eigenvalues)}")

def checkMatrixisPd(A):
	checkMatrixisSymmetric(A)
	eigenvalues=np.linalg.eigvals(A);
	verify(np.all(eigenvalues > 0.0), f"Matrix is not PD, min eigenvalue is {np.amin(eigenvalues)}")

def isMatrixSingular(A):
	return (np.linalg.matrix_rank(A) < self.E.shape[0])

#Taken from https://gist.github.com/sgsfak/77a1c08ac8a9b0af77393b24e44c9547
#Compute the Reduced Row Echelon Form (RREF) in Python
def rref(B, tol=1e-8):
	A = B.copy()
	rows, cols = A.shape
	r = 0
	pivots_pos = []
	row_exchanges = np.arange(rows)
	for c in range(cols):

		## Find the pivot row:
		pivot = np.argmax (np.abs (A[r:rows,c])) + r
		m = np.abs(A[pivot, c])
		if m <= tol:
			## Skip column c, making sure the approximately zero terms are
			## actually zero.
			A[r:rows, c] = np.zeros(rows-r)
		else:
			## keep track of bound variables
			pivots_pos.append((r,c))

			if pivot != r:
				## Swap current row and pivot row
				A[[pivot, r], c:cols] = A[[r, pivot], c:cols]
				row_exchanges[[pivot,r]] = row_exchanges[[r,pivot]]

			## Normalize pivot row
			A[r, c:cols] = A[r, c:cols] / A[r, c];

			## Eliminate the current column
			v = A[r, c:cols]
			## Above (before row r):
			if r > 0:
				ridx_above = np.arange(r)
				A[ridx_above, c:cols] = A[ridx_above, c:cols] - np.outer(v, A[ridx_above, c]).T
			## Below (after row r):
			if r < rows-1:
				ridx_below = np.arange(r+1,rows)
				A[ridx_below, c:cols] = A[ridx_below, c:cols] - np.outer(v, A[ridx_below, c]).T
			r += 1
		## Check if done
		if r == rows:
			break;
	return (A, pivots_pos, row_exchanges)

# Input is the matrices (A,b) that define the system Ax=b
# Ouput is the matrices (A',b') such that A'x=b' has the same solutions as Ax=b, and (A',b') have the smallest possible number of rows
# See https://mathoverflow.net/a/48867  and the row echelon form in https://en.wikipedia.org/wiki/Gaussian_elimination#General_algorithm_to_compute_ranks_and_bases
def removeRedundantEquationsFromEqualitySystem(A, b):

	A_b=np.concatenate((A, b), axis=1)

	A_b_new, pivots_pos, row_exchanges = rref(A_b)

	##### Now remove the rows that are zero
	rows_that_are_zero=[]
	for i in range(A_b_new.shape[0]):
		if(np.linalg.norm(A_b_new[i,:])<1e-7):
			rows_that_are_zero.append(i)

	A_b_new = np.delete(A_b_new, rows_that_are_zero, axis=0)
	#####################################

	##### Now get A_new and b_new from A_b_new
	A_new=A_b_new[:,:-1].reshape((-1, A.shape[1]))
	b_new=A_b_new[:,-1].reshape((-1, 1))
	#####################################

	if(A_b_new.shape[0]>0):
		assert np.linalg.matrix_rank(A_b_new)==A_b_new.shape[0]

	return A_new, b_new



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

	#Each column of V is a vertex
	#Each column of R is a ray

	return V, R




