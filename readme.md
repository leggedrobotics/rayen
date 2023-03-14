```
https://github.com/jtorde/linear_constraints_NN
git submodule update --init --recursive
```
--> The module can be at the end of the NN (in the output), but also in between

--> The trick of dividing by the infinity norm to enforce the box constraints does not work when the polyhedron is not a cone (if it is cone it works because it still obeys the conic constraints due to pure scaling)

--> remove the deep_panther dependency (it should only depend on minvo) & copy MyClampedSpline file 
