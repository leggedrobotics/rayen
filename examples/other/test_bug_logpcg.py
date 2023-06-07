import numpy as np
import torch
import scipy

# def using_power_iteration(C):
	  
#     b_k = np.random.rand(C.shape[0],1)
	  
#     tol = 1e-6
#     max_iter = 100
	  
#     for i in range(max_iter):
#         b_k1 = C @ b_k
#         b_k1 = b_k1 / np.linalg.norm(b_k1)

#         lam=(b_k1.T @ b_k)/(b_k.T @ b_k)
#         print(lam)
#         # x = C @ x / np.linalg.norm(C @ x)
#         # lam = (x.T @ C @ x) / (x.T @ x)
#         # # print(lam)
#         # if np.abs(lam - lam_prev) < tol:
#         #     break
#         # lam_prev = lam
	  
#     return lam

def eigenvalue(A, v):
	Av = A.dot(v)
	return v.dot(Av)

def power_iteration(A):
	n, d = A.shape

	v = np.ones(d) / np.sqrt(d)
	ev = eigenvalue(A, v)

	while True:
		Av = A.dot(v)
		v_new = Av / np.linalg.norm(Av)

		ev_new = eigenvalue(A, v_new) #v_new.dot(A.dot(v_new)) #
		if np.abs(ev - ev_new) < 1e-12:
			break

		v = v_new
		ev = ev_new

	return ev_new #, v_new


dim=3

while True:
	print("\n\n-------")

	#Random generation of matrices A and B
	tmp = np.random.uniform(-1, 1, (dim, dim))
	A=(tmp+tmp.T)/2.0  #A is symmetric by construction

	tmp = np.random.uniform(-1, 1, (dim, dim))
	B = np.dot(tmp, tmp.transpose()) + np.eye(dim) #B is symmetric positive definite by construction

	print(f"A={A}\n")
	print(f"B={B}\n")


	X=np.random.rand(A.shape[0], 1) #Initial guess for the eigenvector

	#### USING lobpcg
	lambda_lobpcg, _ = torch.lobpcg(A=torch.from_numpy(A), k=1, B=torch.from_numpy(B), niter=-1, X=torch.from_numpy(X), tol=1e-12)
	lambda_lobpcg = lambda_lobpcg.item()
	print(f"lambda_lobpcg={lambda_lobpcg}")

	# #### USING Scipy
	lambda_scipy, _ =scipy.sparse.linalg.lobpcg(A=A, B=B, X=X, maxiter=10000)
	lambda_scipy=lambda_scipy[0]
	print(f"lambda_scipy={lambda_scipy}")

	#### USING normal eigendecomposition
	all_lambdas, _=np.linalg.eig(np.linalg.inv(B)@A); 
	# print(f"all_lambdas={all_lambdas}")
	lambda_eig=np.max(all_lambdas)
	print(f"lambda_eig={lambda_eig}")

	#### USING Power iteration
	lambda_power_iter=power_iteration(np.linalg.inv(B)@A)
	print(f"lambda_power_iter={lambda_power_iter}")

	assert abs(lambda_lobpcg-lambda_eig)<1e-6
	assert abs(lambda_scipy-lambda_eig)<1e-6
	assert abs(lambda_power_iter-lambda_eig)<1e-6





