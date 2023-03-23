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

--> The module can be at the end of the NN (in the output), but also in between

--> The trick of dividing by the infinity norm to enforce the box constraints does not work when the polyhedron is not a cone (if it is cone it works because it still obeys the conic constraints due to pure scaling)


--> Constrained optimization on matrices: https://github.com/lezcano/geotorch

--> https://arxiv.org/pdf/1909.07374.pdf

--> https://arxiv.org/pdf/1706.02025.pdf

--> https://arxiv.org/pdf/2211.01340.pdf

--> https://arxiv.org/pdf/2002.01600v1.pdf (creo que solo vale para equality constraints)

--> hopfield linear constraints

--> Use the kernel trick to extend it to nonlinearities?

--> Check the case when the only constraint is just a plane

--> Examples:

-------> End effector on a plane
-------> In-painting: https://www.cvxpy.org/examples/applications/tv_inpainting.html (but here the only constraint you have is that each pixel needs to be \in [0,255]). See also https://github.com/jhultman/image-inpainting/blob/master/src/inpainting.py#L69
-------> Drone racing (gate positions are known, initial position is known, goal in area. Want to trade-off smoothness vs )

star convex set idea