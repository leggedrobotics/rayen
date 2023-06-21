import numpy as np
import utils
from constraint_layer import ConstraintLayer, CostComputer
import torch
A1=np.array([[-1.0, 0],
			 [0.0, -1.0],
			 [1.0, 0.0],
			 [0.0, 1.0]]);

b1=np.array([[0.0],
			[0.0],
			[1.0],
			[1.0]])


lc=utils.LinearConstraint(A1, b1, None, None)

cs=utils.convexConstraints(lc=lc, qcs=[], socs=[], lmic=None)

constraint_layer=ConstraintLayer(cs, input_dim=2, method='UU', create_map=False) 


y=torch.Tensor([
    [[5.0],[0.5]],
    [[5.0],[0.5]]
	])
y=y.double()

y_predicted=constraint_layer(y)

print(y_predicted)


cost_computer=CostComputer(cs)


soft_cost=cost_computer.getSumSoftCostAllSamples(y_predicted)

projected_cost=np.sum(np.apply_along_axis(cs.getViolation,axis=1, arr=y_predicted.cpu().numpy())).item()/y.shape[0]


print(f"Soft cost={soft_cost}")
print(f"Projected cost={projected_cost}")



# x=tensor([[[0.2848],
#          [0.1653],
#          [0.1750]]], device='cuda:0')
# 338
# y_to_be_projected=[[ 8.24753598]
#  [-0.3411386 ]
#  [ 8.22857904]
#  [-0.40048401]
#  [ 8.20338498]
#  [-0.42033985]
#  [ 8.55192249]
#  [ 0.03189726]
#  [10.92285101]
#  [ 2.89443184]
#  [15.36493899]
#  [ 9.68558388]
#  [18.75277026]
#  [10.72327992]
#  [20.55312588]
#  [ 8.65413096]
#  [21.41829213]
#  [ 7.19982242]]
# Projection distance=290.47933174765967
# optimal
# y_to_be_projected=[[ 0.87061467]
#  [ 5.88906773]
#  [ 0.87061467]
#  [ 5.88906773]
#  [ 0.87061467]
#  [ 5.88906773]
#  [ 6.46999776]
#  [ 2.56039083]
#  [11.45420182]
#  [ 2.37858093]
#  [15.24621285]
#  [ 9.81629311]
#  [20.34782588]
#  [ 8.84582365]
#  [20.34782588]
#  [ 8.84582365]
#  [20.34782588]
#  [ 8.84582365]]
# Projection distance=9.08084120363035e-10
# optimal
