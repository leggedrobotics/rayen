# --------------------------------------------------------------------------
# Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
# See LICENSE file for the license information
# -------------------------------------------------------------------------- 

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