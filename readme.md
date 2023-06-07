# RAYEN: Imposition of Hard Convex Constraints on Neural Networks #

![](./rayen.png)

# Installation

```bash
pip install rayen
```

If you want to do an editable install, you can do:
```bash
git clone https://github.com/leggedrobotics/rayen.git
cd rayen && pip install -e .
```

# Usage

A minimal example (with only linear and quadratic constraints) is as follows:

```python
import torch
import numpy as np
from rayen import constraints, constraint_module

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
qcs = [constraints.ConvexQuadraticConstraint(P, q, r)]

#Add SOC and LMI (SDP) constraints here if needed
# ...

cs = constraints.ConvexConstraints(lc=lc, qcs=qcs, socs=[], sdpc=None)

model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3, 64), 
			    torch.nn.ReLU(),    torch.nn.Linear(64, 64),
			    torch.nn.ReLU(),    torch.nn.Linear(64, 64),
			    constraint_module.ConstraintModule(cs, input_dim=64, create_map=True)) 

x_batched = torch.Tensor(500, 3, 1).uniform_(-1.0, 1.0)
y_batched = model(x_batched)

#Each element of y_batched is guaranteed to satisfy the constraints

# loss = ...      # y_batched can be used here
# loss.backward() # Backpropagate
```

You can choose the method used. These are the methods (please see the paper for details):


Method | Linear | Quadratic | SOC | LMI
:------------ | :-------------: | :-------------: | :-------------: | :-------------: 
**UU** | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:
**UP** | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:
**PP** | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:
**DC3** | :heavy_check_mark: |  :heavy_check_mark: | :large_orange_diamond: | :large_orange_diamond:
**Bar** | :heavy_check_mark: |  :x: | :x: | :x:
**RAYEN** | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:

where :heavy_check_mark: denotes supported by the algorithm, :x: denotes not supported by the algorithm, and :large_orange_diamond: denotes supported by the algorithm but not implemented yet.

$$\boldsymbol{A}_{1}\boldsymbol{y}\le\boldsymbol{b}_{1}$$

$$\boldsymbol{A}$$

$$\mathbf{A}$$

$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$

# More examples

There are more complex examples in the `examples` folder. If you want to run these examples: 

```bash
sudo apt-get install git-lfs
git clone https://github.com/leggedrobotics/rayen.git
cd rayen
git lfs install 
git submodule init && git submodule update --init --recursive
pip install examples/requirement_examples.txt
```

Some of these examples use (or can use) [Gurobi Optimizer](https://www.gurobi.com/products/gurobi-optimizer/). Once installed (following the instructions in the previous link) you can test the installation typing `gurobi.sh` in the terminal. You will also need this package:
```
pip install gurobipy
```

To generate the data used in the examples (the files `corridor_dim2.mat` and `corridor_dim3.mat`, you need to run the Matlab file `traj_planning_in_corridor.m`. This file requires Casadi (its interface with Gurobi) to be installed. You can do this using the instructions below:

<details>
  <summary> <b>Casadi installation instructions (optional dependency)</b></summary>

```bash
#IPOPT stuff
sudo apt-get install gcc g++ gfortran git cmake liblapack-dev pkg-config --install-recommends
sudo apt-get install coinor-libipopt1v5 coinor-libipopt-dev

#SWIG stuff
sudo apt-get remove swig swig3.0 swig4.0 #If you don't do this, the compilation of casadi may fail with the error "swig error : Unrecognized option -matlab"
mkdir ~/installations && cd ~/installations
git clone https://github.com/jaeandersson/swig
cd swig
git checkout -b matlab-customdoc origin/matlab-customdoc        
sh autogen.sh
sudo apt-get install gcc-7 g++-7 bison byacc
sudo apt-get install libpcre3 libpcre3-dev
./configure CXX=g++-7 CC=gcc-7            
make
sudo make install

#CASADI stuff
cd ~/installations && mkdir casadi && cd casadi
git clone https://github.com/casadi/casadi
cd casadi/cmake && wget https://github.com/leggedrobotics/rayen/raw/master/examples/other/FindGurobi.cmake #This ones works for higher versions of Gurobi
cd ..
#cd build && make clean && cd .. && rm -rf build #Only if you want to clean any previous installation/compilation 
mkdir build && cd build
cmake . -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DWITH_MATLAB=ON -DWITH_PYTHON=ON -DWITH_DEEPBIND=ON -DWITH_GUROBI=ON ..
#You may need to run the command above twice until the output says that `Ipopt` has been detected (although `IPOPT` is also being detected when you run it for the first time)
make -j20
sudo make install

```
</details>
