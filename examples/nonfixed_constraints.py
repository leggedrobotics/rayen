# --------------------------------------------------------------------------
# Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich
# See LICENSE file for the license information
# --------------------------------------------------------------------------

# Synthetic experiment with non-fixed constraints (Section 7.1 of the paper):
# The linear inequality constraints A1*y<=b1 and the convex quadratic constraints 0.5*y'*P_i*y + q_i'*y + r_i<=0
# change in each forward pass (they are different for each sample and for each batch).
# Following Section 7.1, they are generated so that b1>0, r_i<0, and P_i=V_i'*V_i. Hence, y0=0 (i.e., z0=0) is always
# a strictly feasible point, and the offline phase is not needed.
# The network receives as input the constraints and a target point y_target, and it is trained to minimize ||y-y_target||^2
# Run it with "python -O nonfixed_constraints.py". The -O flag removes the asserts (as when testing in scripts/run.sh), which would otherwise
# add GPU synchronizations to the measured computation times. See https://docs.python.org/3/using/cmdline.html#cmdoption-O

import time
import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp

import fixpath #Following this example: https://github.com/tartley/colorama/blob/master/demos/demo01.py
from rayen import constraints, constraint_module, utils

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

if __debug__:
	utils.printInBoldRed("Asserts are enabled. Run this script with 'python -O nonfixed_constraints.py' to obtain accurate computation times")

k=3              #Dimension of y
num_lin=6        #Number of linear inequality constraints (rows of A1)
num_quad=2       #Number of convex quadratic constraints
batch_size=256
num_iterations=100000
num_samples_test=1000
device=torch.device('cuda:0')

def generateRandomConstraints(num_samples):
	A1=torch.empty(num_samples, num_lin, k, device=device).uniform_(-1.0, 1.0)
	b1=torch.empty(num_samples, num_lin, 1, device=device).uniform_(0.1, 1.0)               # b1>0 (Eq. nonfixed_b1)
	V=torch.empty(num_samples, num_quad, k, k, device=device).uniform_(-1.0, 1.0)
	P=V.transpose(2,3)@V                                                                    # P_i=V_i'*V_i is PSD
	q=torch.empty(num_samples, num_quad, k, 1, device=device).uniform_(-1.0, 1.0)
	r=torch.empty(num_samples, num_quad, 1, 1, device=device).uniform_(-1.0, -0.1)         # r_i<0 (Eq. nonfixed_ri)
	y_target=torch.empty(num_samples, k, 1, device=device).uniform_(-2.0, 2.0)
	return A1, b1, P, q, r, y_target

def getInputNetwork(A1, b1, P, q, r, y_target):
	return torch.cat((A1.flatten(1), b1.flatten(1), P.flatten(1), q.flatten(1), r.flatten(1), y_target.flatten(1)), dim=1)

#Updates the quantities that RAYEN needs online (D, phi_i, delta_i, see Section 4) for the current constraints (one set of constraints per sample)
def updateConstraints(layer, A1, b1, P, q, r):
	#With y0=0 and no equality constraints: A_p=A1*N, b_p=b1, z0=0 --> D=A_p/b_p
	layer.D=(A1@layer.NA_E)/b1
	#With y0=0, the expressions of phi and delta used in ConstraintModule become:
	layer.all_phi=(-q.transpose(2,3)/(2*r)).transpose(0,1)                                # [num_quad, num_samples, 1, k]
	layer.all_delta=((q@q.transpose(2,3) - 2*r*P)/(4*torch.square(r))).transpose(0,1)     # [num_quad, num_samples, k, k]

#Create the layer using a nominal set of constraints (whose values are then overwritten in each forward pass by updateConstraints())
A1_nom, b1_nom, P_nom, q_nom, r_nom, _ = [x[0].cpu().numpy() for x in generateRandomConstraints(1)]
lc=constraints.LinearConstraint(A1=A1_nom, b1=b1_nom, A2=None, b2=None)
qcs=[constraints.ConvexQuadraticConstraint(P=P_nom[i], q=q_nom[i], r=r_nom[i]) for i in range(num_quad)]
cs=constraints.ConvexConstraints(lc=lc, qcs=qcs, socs=[], lmic=None, y0=np.zeros((k,1)), do_preprocessing_linear=False)

numel_input=num_lin*k + num_lin + num_quad*(k*k + k + 1) + k
layer=constraint_module.ConstraintModule(cs, input_dim=512, method='RAYEN', create_map=True)
model=nn.Sequential(nn.Linear(numel_input, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), layer).to(device)
optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations)

########################################### TRAINING (new random constraints in each iteration)
model.train()
for it in range(num_iterations):
	A1, b1, P, q, r, y_target = generateRandomConstraints(batch_size)
	updateConstraints(layer, A1, b1, P, q, r)
	y=model(getInputNetwork(A1, b1, P, q, r, y_target))
	loss=torch.mean(torch.sum(torch.square(y-y_target), dim=1))
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	scheduler.step()
	if(it%10000==0):
		print(f"Iteration {it}: loss={loss.item():.4f}")

########################################### TESTING
model.eval()
A1, b1, P, q, r, y_target = generateRandomConstraints(num_samples_test)
x=getInputNetwork(A1, b1, P, q, r, y_target)

with torch.no_grad():
	updateConstraints(layer, A1, b1, P, q, r); model(x) #Warm up the GPU (for a better estimate of the computation time)
	torch.cuda.synchronize() #Wait for the warm-up to finish, so that both measurements below start with the GPU idle

	cuda_timer=utils.CudaTimer()
	cuda_timer.start()
	updateConstraints(layer, A1, b1, P, q, r)
	time_update_s=cuda_timer.endAndGetTimeSeconds()

	cuda_timer.start()
	y=model(x)
	time_forward_s=cuda_timer.endAndGetTimeSeconds()

loss_rayen=torch.sum(torch.square(y-y_target), dim=1).squeeze()

A1, b1, P, q, r, y_target, y = [tmp.cpu().numpy() for tmp in (A1, b1, P, q, r, y_target, y)]

#Violation of the constraints, computed as in main.py (Section 6.1 of the paper): squared distance from the output of the network
#to the closest point of the feasible set (i.e., to its projection onto the feasible set), averaged over all the samples
violation=0.0
for i in range(num_samples_test):
	lc_i=constraints.LinearConstraint(A1=A1[i], b1=b1[i], A2=None, b2=None)
	qcs_i=[constraints.ConvexQuadraticConstraint(P=P[i,j], q=q[i,j], r=r[i,j]) for j in range(num_quad)]
	cs_i=constraints.ConvexConstraints(lc=lc_i, qcs=qcs_i, socs=[], lmic=None, y0=np.zeros((k,1)), do_preprocessing_linear=False)
	violation+=cs_i.getViolation(y[i])/num_samples_test

#Globally-optimal solution (projection of y_target onto the feasible set), obtained with a convex solver
loss_opt=[]
time_opt_s=0.0      #Solve time reported by Gurobi
time_opt_wall_s=0.0 #Also includes the time CVXPY needs to build the problem
for i in range(num_samples_test):
	y_opt=cp.Variable((k,1))
	cons=[A1[i]@y_opt<=b1[i]] + [0.5*cp.quad_form(y_opt, P[i,j]) + q[i,j].T@y_opt + r[i,j]<=0 for j in range(num_quad)]
	prob=cp.Problem(cp.Minimize(cp.sum_squares(y_opt-y_target[i])), cons)
	start=time.perf_counter()
	prob.solve(solver=cp.GUROBI)
	time_opt_wall_s+=time.perf_counter()-start
	time_opt_s+=prob.solver_stats.solve_time
	loss_opt.append(prob.value)

print("==========================================")
print(f"Normalized loss (RAYEN):              {torch.mean(loss_rayen).item()/np.mean(loss_opt):.4f}")
print(f"Violation (RAYEN):                    {violation:.2e}")
print(f"Time update constraints (per sample): {1e6*time_update_s/num_samples_test:.3f} us")
print(f"Time forward pass (per sample):       {1e6*time_forward_s/num_samples_test:.3f} us")
print(f"Solve time of Gurobi (per sample):    {1e6*time_opt_s/num_samples_test:.1f} us")
print(f"Time CVXPY+Gurobi (per sample):       {1e6*time_opt_wall_s/num_samples_test:.1f} us")
