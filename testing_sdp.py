# Import packages.
import cvxpy as cp
import numpy as np
from numpy import linalg as LA
import utils

dim=4

# Generate a random SDP.
tmp = np.random.rand(dim, dim)
H = np.dot(tmp, tmp.transpose()) #H is psd by construction

tmp = np.random.rand(dim, dim)
M=(tmp+tmp.T)/2.0  #M is symmetric by construction

# tmp = np.random.rand(dim, dim)
# M = np.dot(tmp, tmp.transpose()) #H is psd by construction


Hinv=np.linalg.inv(H)
Minv=np.linalg.inv(M)

print(f"H={H}\n")
print(f"M={M}\n")
print("-------")
print("-------")

# lam = cp.Variable()
# constraints = [(H+lam*M) >> 0]
# prob = cp.Problem(cp.Maximize(lam), constraints)
# prob.solve(solver=cp.SCS, verbose=True, eps=1e-7)

kappa = cp.Variable()
constraints = [(kappa*H+M) >> 0, kappa>=0]

prob = cp.Problem(cp.Minimize(kappa), constraints)
prob.solve(solver=cp.SCS, verbose=True, eps=1e-7)

if(prob.status=='unbounded'):
    utils.printInBoldRed("Unbounded!!!!") #When kappa is the decision variable, there is no way the result is unbounded
else:
    utils.printInBoldBlue(f"kappa={kappa.value}")


print("-------")
print("Eigen Decomposition of H")

wH, vH = LA.eig(H)
print(f"Eigenvalues={wH}")
print(f"Eigenvectors=\n{vH}")

print("-------")
print("Eigen Decomposition of M")

wM, vM = LA.eig(M)
print(f"Eigenvalues={wM}")
print(f"Eigenvectors=\n{vM}\n")

# for i in range(wH.shape[0]):
#     for j in range(wM.shape[0]):
#         print(wH[i]/wM[j])
#         print(wM[j]/wH[i])
#         print(wH[i]*wM[j])


#This should be zero
# print(np.linalg.det(H+lam.value*M))

# print(np.linalg.det(Minv@H+lam.value*np.eye(2)))

print("======")
print(Hinv)

all_kappas, _=np.linalg.eig(-Hinv@M); #Be careful because Hinv@M is NOT symmetric (Hinv and M is)

print("--------------")
print(all_kappas) #El largest element is the solution
kappa=np.max(all_kappas)
print(f"kappa={kappa}")
# print(f"lambda={1/kappa}")
print("--------------")


for i in range(all_kappas.shape[0]):
    tmp=all_kappas[i]

    # print(np.linalg.det(-Hinv@M-tmp*np.eye(dim)))
    print(f"tmp={tmp}")
    eigenvals,_=np.linalg.eig(tmp*H+M)
    print(eigenvals)


# beta = cp.Variable()
# constraints = [H >> beta*M]

# prob = cp.Problem(cp.Minimize(beta), constraints)
# prob.solve()
# print(f"beta={beta.value}")

# Hinv=np.linalg.inv(H)
# constraints = [(np.eye(2)+lam*Hinv@M) >> 0]
# prob = cp.Problem(cp.Maximize(lam), constraints)
# prob.solve()
# print(f"lam={lam.value}")

# L = np.linalg.cholesky(H)
# # print(L@L.T-H) #good

# Y=np.linalg.inv(L)@M@(np.linalg.inv(L).T)

# constraints = [(Y+lam*np.eye(2)) >> 0]

# prob = cp.Problem(cp.Maximize(lam), constraints)
# prob.solve()
# print(f"lam={lam.value}")


