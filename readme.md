# RAYEN: Imposition of Hard Convex Constraints on Neural Networks #

![](./rayen.png)

A minimal example (with only linear and quadratic constraints) is as follows:

```python
...
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


To install this package:
```bash
git clone https://github.com/leggedrobotics/rayen.git
cd rayen
pip install -e .
```

If you wanna run the examples, you need to 

```bash
pip install examples/requirement_examples.txt
```