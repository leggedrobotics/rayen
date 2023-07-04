# --------------------------------------------------------------------------
# Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich 
# See LICENSE file for the license information
# -------------------------------------------------------------------------- 

import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import numpy as np
import scipy
import os
import time

from examples_sets import getExample
import utils_examples

import fixpath #Following this example: https://github.com/tartley/colorama/blob/master/demos/demo01.py
from rayen import constraints, constraint_module, utils

method='RAYEN'
index_example=13


fig = plt.figure()
fig.suptitle(method, fontsize=10)

tmp=getExample(index_example)

constraint=constraints.ConvexConstraints(lc=tmp.lc, qcs=tmp.qcs, socs=tmp.socs, lmic=tmp.lmic, y0=np.array([[0.5], [0.0], [0.8]]))

ax = fig.add_subplot(1,1,1, projection="3d")

my_layer=constraint_module.ConstraintModule(constraint, method=method, create_map=False)

num_samples=500 #12000
v_batched_x=torch.Tensor(num_samples, 1, 1).uniform_(-0.5, 0.6)
v_batched_y=torch.Tensor(num_samples, 1, 1).uniform_(-0.55, 0.47)
v_batched_z=torch.Tensor(num_samples, 1, 1).uniform_(-0.4, 0.3)
v_batched = torch.cat((v_batched_x, v_batched_y, v_batched_z),1)

my_layer.eval() #This changes the self.training variable of the module

time_start=time.time()
result=my_layer(v_batched)
total_time_per_sample= (time.time()-time_start)/num_samples

result=result.detach().numpy();

y0=my_layer.gety0();


ax.scatter(y0[0,0], y0[1,0], y0[2,0],color='r',s=500)
ax.scatter(result[:,0,0], result[:,1,0], result[:,2,0])


my_dict=constraint.getDataAsDict();
my_dict["result"]=result
my_dict["total_time_per_sample"]=total_time_per_sample
my_dict["y0"]=constraint.y0
my_dict["v"]=v_batched.detach().numpy()
directory='./first_figure'
if not os.path.exists(directory):
	os.makedirs(directory)
scipy.io.savemat(directory+'/first_figure.mat', my_dict)

utils.printInBoldBlue(f"Example {index_example}, total_time_per_sample={total_time_per_sample}")

plt.show()