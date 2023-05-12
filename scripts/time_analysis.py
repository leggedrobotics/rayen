import numpy as np
import torch
import time
from joblib import Parallel, delayed
import pandas as pd
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))  #See first comment of this answer: https://stackoverflow.com/a/11158224

import utils
from constraint_layer import ConstraintLayer
import constraints

path="./results/"
if not os.path.exists(path):
   os.makedirs(path)

torch.set_default_dtype(torch.float64)

def getTime_sMethod(cs, num_samples):
	print("Calling constructor for the layer...")
	my_layer=ConstraintLayer(cs, method="walker_1", create_map=False)
	print("Called")

	

	numel_output_mapper=my_layer.getDimAfterMap()
	x_batched=torch.Tensor(num_samples, numel_output_mapper, 1).uniform_(-1.0, 1.0)
	print("Calling method...")
	
	cuda_timer=utils.CudaTimer()
	cuda_timer.start()
	result=my_layer(x_batched)
	total_s_per_sample= cuda_timer.endAndGetTimeSeconds()/num_samples
	print("Called")

	print(f"total_s_per_sample={total_s_per_sample} s")
	return total_s_per_sample

# all_k = [1, 10, 100, 1000, 10000]


num_samples=2000

# ########################################### Linear CONSTRAINTS
times_lin = []

all_k = [1, 10, 100, 1000, 2000, 3000, 4000, 5000, 10000]
all_r_A1 = [1, 10, 100, 500, 1000, 2000, 3000] 

for r_A1 in all_r_A1:
	for k in all_k:
		utils.printInBoldRed(f"r_A1 = {r_A1}, k={k}")
		A1=np.random.uniform(low=-1.0, high=1.0, size=(r_A1,k))
		b1=np.random.uniform(low= 0.1, high=1.0, size=(r_A1,1)) #In this way, y=0 is always a feasible solution

		lc=constraints.LinearConstraint(A1=A1, b1=b1, A2=None, b2=None);
		cs=constraints.convexConstraints(lc=lc, qcs=[], socs=[], sdpc=None, y0=np.zeros((k,1)), do_preprocessing_linear=False)
		total_s=getTime_sMethod(cs, num_samples)
		times_lin.append({'k': k,'r_A1': r_A1,'Time': total_s})

times_lin=pd.DataFrame(times_lin)
# utils.savepickle(times_lin, path+"times_lin.pkl")
times_lin.to_csv(path+"times_lin.csv",index=False)


# ########################################### QP CONSTRAINTS
times_qp = []
all_eta = [1, 10, 50, 100, 500, 1000] #np.power(10, np.array([0,1,2,3]))
all_k = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for eta in all_eta:
	for k in all_k:
		utils.printInBoldRed(f"QP constraints, eta = {eta}, k={k}")

		qcs=[]

		def createRandomQPConstraint(i):
			tmp=np.random.uniform(low=-1.0, high=1.0, size=(k,k))
			# tmp=(torch.empty((k,k), device="cuda:0").uniform_(-1.0, 1.0)).cpu().numpy() #

			P=tmp@tmp.T #To make sure it's a (symmetric) PSD matrix
			q=np.random.uniform(low=-1.0, high=1.0, size=(k,1))
			r=np.random.uniform(low=-1.0, high=0.0, size=(1,1))
			qc=constraints.convexQuadraticConstraint(P=P, q=q, r=r, do_checks_P=False);
			return qc

		print("Creating random constraints")
		qcs = Parallel(n_jobs=15)(delayed(createRandomQPConstraint)(i) for i in range(eta))
		print("Created")

		assert len(qcs)==eta
		
		cs=constraints.convexConstraints(lc=None, qcs=qcs, socs=[], sdpc=None, y0=np.zeros((k,1)))
		total_s=getTime_sMethod(cs, num_samples)
		times_qp.append({'k': k,'eta': eta,'Time': total_s})

times_qp=pd.DataFrame(times_qp)
# utils.savepickle(times_qp, path+"times_qp.pkl")
times_qp.to_csv(path+"times_qp.csv",index=False)


########################################### SOC CONSTRAINTS
times_soc = []
all_r_M = [10, 100, 200, 300]
all_mu = [10, 100, 300, 500]
all_k = [10, 100, 500, 1000] 

for r_M in all_r_M:
	for mu in all_mu:
		for k in all_k:
			utils.printInBoldRed(f"r_M = {r_M}, mu={mu}, k={k}")

			def createRandomSOCConstraint(i):
				M = np.random.uniform(low=-1.0, high=1.0, size=(r_M,k))
				s = np.random.uniform(low=-1.0, high=1.0, size=(r_M,1))
				c = np.random.uniform(low=-1.0, high=1.0, size=(k,1))
				d = np.linalg.norm(s)+np.array([[0.5]]) #This ensures that 0 will always be a point in the interior of the feasible set
				return M, s, c, d

			print("Creating random constraints")
			all_Mscd = Parallel(n_jobs=15)(delayed(createRandomSOCConstraint)(i) for i in range(k))
			print("Created")

			socs=[]
			for Mscd in all_Mscd:
				socs.append(constraints.SOCConstraint(Mscd[0], Mscd[1], Mscd[2], Mscd[3]))

			print("Creating cs")
			cs=constraints.convexConstraints(lc=None, qcs=[], socs=socs, sdpc=None, y0=np.zeros((k,1)))
			print("Created")

			total_s=getTime_sMethod(cs, num_samples)

			times_soc.append({'k': k,'r_M': r_M,'mu': mu,'Time': total_s})

times_soc=pd.DataFrame(times_soc)
# utils.savepickle(times_soc, path+"times_soc.pkl")
times_soc.to_csv(path+"times_soc.csv",index=False)

print(times_soc)


########################################### SDP CONSTRAINTS
times_sdp = []
all_r_F = [10, 100, 200, 300]
all_k = [100, 500, 1000, 2000, 5000, 7000, 10000]

for r_F in all_r_F:
	for k in all_k:
		utils.printInBoldRed(f"r_F = {r_F}, k={k}")

		def createRandomSymmetricMatrix(i):
			tmp=np.random.uniform(low=-1.0, high=1.0, size=(r_F,r_F))
			return (tmp + tmp.T)/2 #To ensure that it is symmetric

		print("Creating random constraints")
		all_F = Parallel(n_jobs=15)(delayed(createRandomSymmetricMatrix)(i) for i in range(k))

		tmp = np.random.uniform(-1, 1, (r_F, r_F))
		F_k = np.dot(tmp, tmp.transpose()) + 0.5*np.eye(r_F) #F_k is symmetric positive definite by construction
		all_F.append(F_k)
		print("Created")

		print("Creating sdpc")
		sdpc=constraints.SDPConstraint(all_F)
		print("Created sdpc")

		print("Creating cs")
		cs=constraints.convexConstraints(lc=None, qcs=[], socs=[], sdpc=sdpc, y0=np.zeros((k,1)))
		print("Created")

		total_s=getTime_sMethod(cs, num_samples)

		times_sdp.append({'k': k,'r_F': r_F,'Time': total_s})

times_sdp=pd.DataFrame(times_sdp)
# utils.savepickle(times_sdp, path+"times_sdp.pkl")
times_sdp.to_csv(path+"times_sdp.csv",index=False)

print(times_sdp)