import scipy
import numpy as np
from scipy.linalg import null_space

Aeq = np.array([[1, 1, 1]])
N = scipy.linalg.null_space(Aeq) #Each column is a vector of the basis

print(N) #This matrix is 3x2