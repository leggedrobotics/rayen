


A minimal example (with only linear and quadratic constraints) is as follows:

```python
import torch
import numpy as np
import constraints, constraint_module

#Linear constraints: A 3D Cube and a plane in 3D
A1 = np.array([ [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]);
b1 = np.array([[1.0], [1.0], [1.0], [0], [0], [0]])
A2 = np.array([[1.0, 1.0, 1.0]]);
b2 = np.array([[1.0]]);
lc=constraints.LinearConstraint(A1, b1, A2, b2)

#Quadratic constraints: A Sphere 
P = np.array([[3.125 , 0.0   , 0.0  ], [0.0   , 3.125, 0.0   ], [0.0   , 0.   , 3.125]])
q = np.array([[0.0],[0.0],[0.0]])
r = np.array([[-1.0]])
qcs = [constraints.convexQuadraticConstraint(P, q, r)]

#Add SOC and LMI (SDP) constraints here if needed
# ...

cs = constraints.convexConstraints(lc=lc, qcs=qcs, socs=[], sdpc=None)

model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3, 64), 
		            torch.nn.ReLU(),    torch.nn.Linear(64, 64),
		            torch.nn.ReLU(),    torch.nn.Linear(64, 64),
			    constraint_module.ConstraintModule(cs, input_dim=64, create_map=True)) 

x_batched = torch.Tensor(500, 3, 1).uniform_(-1.0, 1.0)
y_batched = model(x_batched)

#Each element of y_batched is guaranteed to satisfy the constraints

loss = ...      # y_batched can be used here
loss.backward() # Backpropagate
```


```
https://github.com/jtorde/linear_constraints_NN
git submodule update --init --recursive
```

To install Casadi, with Gurobi, first intall the Gurobi Optimizer (gurobi.sh should work from the terminal), and then follow the steps from deep-panther readme, but change these steps:
```
cd ~/installations/casadi/ 
SUBSTITUTE THE FILE cmake/FindGUROBI.cmake with this other one: https://gitlab.inf.unibe.ch/CGG-public/cmake-library/-/blob/master/finders/FindGurobi.cmake (this ones works for higher versions of Gurobi)
mkdir build && cd build
make clean 
cmake . -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DWITH_MATLAB=ON -DWITH_PYTHON=ON -DWITH_DEEPBIND=ON -DWITH_GUROBI=ON ..
```

For Gurobi to work in python [used in utils.py], you also need to do this (inside the virtual environment):
```
python -m pip install gurobipy
```
