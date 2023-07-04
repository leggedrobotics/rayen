# RAYEN: Imposition of Hard Convex Constraints on Neural Networks #

Paper: Coming soon

This framework allows you to impose convex constraints on the output or latent variable of a Neural Network.
![](./imgs/rayen.png)

![](./imgs/rayen_equations.png)



# Installation

First make sure that you have pip up-to-date:
```bash
pip install --upgrade pip
```

Then, you simply need to do:


```bash
pip install rayen
```

If you want to do an editable install, you can also do:
```bash
git clone https://github.com/leggedrobotics/rayen.git
cd rayen && pip install -e .
```

# Usage

A minimal example is as follows:

```python
import torch
import numpy as np
from rayen import constraints, constraint_module

#Linear constraints
A1 = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]);
b1 = np.array([[1.0], [1.0], [1.0], [0], [0], [0]])
A2 = np.array([[1.0, 1.0, 1.0]]);
b2 = np.array([[1.0]]);
lc=constraints.LinearConstraint(A1, b1, A2, b2) #Set lc to None if there are no linear constraints
                                                #Set A1 and b1 to None if there are no linear inequality constraints
                                                #Set A2 and b2 to None if there are no linear equality constraints

#Quadratic constraints
P = np.array([[3.125,0.0,0.0], [0.0,3.125,0.0], [0.0,0.0,3.125]])
q = np.array([[0.0],[0.0],[0.0]])
r = np.array([[-1.0]])
qcs = [constraints.ConvexQuadraticConstraint(P, q, r)] #Set qcs to [] if there are no quadratic constraints
                                                       #More quadratic constraints can be appended to this list

#SOC constraint
M=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
s=np.array([[0.0],[0.0],[0.0]])
c=np.array([[0.0],[0.0],[1.0]])
d=np.array([[0.0]])
socs = [constraints.SOCConstraint(M, s, c, d)] #Set socs to [] if there are no SOC constraints
                                               #More SOC constraints can be appended to this list

#LMI constraints (semidefinite constraints)
F0=np.array([[1.0, 0.0],[0.0, 0.0]])
F1=np.array([[0.0, 1.0],[1.0, 0.0]])
F2=np.array([[0.0, 0.0],[0.0, 1.0]])
F3=np.array([[0.0, 0.0],[0.0, 0.0]])
lmic=constraints.LMIConstraint([F0, F1, F2, F3]) #Set lmic to None if there are no LMI constraints

#----

cs = constraints.ConvexConstraints(lc=lc, qcs=qcs, socs=socs, lmic=lmic)

model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3, 64),
                            torch.nn.ReLU(),    torch.nn.Linear(64, 64),
                            constraint_module.ConstraintModule(cs, input_dim=64, create_map=True)) 

x_batched = torch.Tensor(500, 3, 1).uniform_(-1.0, 1.0)
y_batched = model(x_batched)

#Each element of y_batched is guaranteed to satisfy the constraints

# loss = ...      # y_batched can be used here
# loss.backward() # Backpropagate
```

These are the methods implemented in this repo (please see the paper for details):




Method | Linear | Quadratic | SOC | LMI
:------------ | :-------------: | :-------------: | :-------------: | :-------------: 
**UU** |    <img src='./imgs/green-tick.png' width='25'>    |     <img src='./imgs/green-tick.png' width='25'>    |    <img src='./imgs/green-tick.png' width='25'>    |    <img src='./imgs/green-tick.png' width='25'>   
**UP** |    <img src='./imgs/green-tick.png' width='25'>    |     <img src='./imgs/green-tick.png' width='25'>    |    <img src='./imgs/green-tick.png' width='25'>    |    <img src='./imgs/green-tick.png' width='25'>   
**PP** |    <img src='./imgs/green-tick.png' width='25'>    |     <img src='./imgs/green-tick.png' width='25'>    |    <img src='./imgs/green-tick.png' width='25'>    |    <img src='./imgs/green-tick.png' width='25'>   
**DC3** |    <img src='./imgs/green-tick.png' width='25'>    |     <img src='./imgs/green-tick.png' width='25'>    |  <img src='./imgs/diamond.svg' width='25'> |  <img src='./imgs/diamond.svg' width='25'>
**Bar** |    <img src='./imgs/green-tick.png' width='25'>    |  <img src='./imgs/red_cross.svg' width='20'> | <img src='./imgs/red_cross.svg' width='20'> | <img src='./imgs/red_cross.svg' width='20'>
**RAYEN** |  <img src='./imgs/green-tick.png' width='25'> |     <img src='./imgs/green-tick.png' width='25'>    |    <img src='./imgs/green-tick.png' width='25'>    |    <img src='./imgs/green-tick.png' width='25'>   

where    <img src='./imgs/green-tick.png' width='20'>    denotes supported by the algorithm and implemented in the code, <img src='./imgs/red_cross.svg' width='15'> denotes not supported by the algorithm, and  <img src='./imgs/diamond.svg' width='20'> denotes supported by the algorithm but not implemented yet. 

You can choose the method to use setting the argument `method` when creating the layer. 

# More examples

There are more complex examples in the `examples` folder. If you want to run these examples: 

```bash
sudo apt-get install git-lfs
git clone https://github.com/leggedrobotics/rayen.git
cd rayen
git lfs install 
git submodule init && git submodule update --init --recursive
pip install -r examples/requirements_examples.txt
```

These are the most important files in the `examples` folder:
* **`test_layer.py`**: This file imposes many different constraints using all the methods shown above. It will create plots similar to the one shown at the beginning of this repo 
* **`time_analysis.py`**: This file obtains the computation time of RAYEN when applied to many different constraints. 
* **`run.h`**: This file will train the networks for all the algorithms used in the paper, and then evaluate them using the testing sets. Depending on the computer to use, this can take ~1 day to run. The datasets that this file uses are stored in the files `corridor_dim2.mat` (Optimization 1 of the paper, which is for a 2D scenario) and `corridor_dim3.mat` (Optimization 2 of the paper, which is for a 3D scenario). These files were generated running the file `traj_planning_in_corridor.m`. This Matlab files requires Casadi (and its interface with Gurobi) to be installed. You can do this using the instructions below

Some of these examples use (or can use) [Gurobi Optimizer](https://www.gurobi.com/products/gurobi-optimizer/). Once installed (following the instructions in the previous link) you can test the installation typing `gurobi.sh` in the terminal. You will also need this package:
```
pip install gurobipy
```

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
