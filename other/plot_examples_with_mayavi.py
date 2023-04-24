import numpy as np
from mayavi import mlab
import utils
import examples_sets
import random   
import copy 
from joblib import Parallel, delayed

index_example=1

cs=examples_sets.getExample(index_example)

# @np.vectorize
def get_conditions(x):
    conditions=[];
    all_is_ineq_condition=[]

    if(cs.has_linear_ineq_constraints):
        for i in range(cs.lc.A1.shape[0]):
            expression=cs.lc.A1[i,:]@x-cs.lc.b1[i,0]
            conditions.append(expression[0])
            all_is_ineq_condition.append(True)

    if(cs.has_linear_eq_constraints):
        for i in range(cs.lc.A2.shape[0]):
            expression=cs.lc.A2[i,:]@x-cs.lc.b2[i,0]
            conditions.append(expression[0])
            all_is_ineq_condition.append(False)


    if(cs.has_quadratic_constraints):
        for qc in cs.qcs:
            expression=0.5*x.T@qc.P@x + qc.q.T@x + qc.r 
            conditions.append(expression[0,0])
            all_is_ineq_condition.append(True)


    if(cs.has_soc_constraints):
        for soc in cs.socs:
            expression=np.linalg.norm(soc.M@x +soc.s) -soc.c.T@x  - soc.d 
            conditions.append(expression[0,0])
            all_is_ineq_condition.append(True)


    return conditions, all_is_ineq_condition

num_points=100j
dist_min=0
dist_max=2
all_x, all_y, all_z = np.mgrid[-dist_min:dist_max:num_points, -dist_min:dist_max:num_points, -dist_min:dist_max:num_points]

num_points=int(num_points.imag)

tmp, all_is_ineq_condition=get_conditions(np.zeros((3,1)))
num_of_conditions=len(tmp)

### Create dummy list of numpy arrays
conditions=[];
for i in range(num_of_conditions):
    zero_tensor=np.zeros((num_points,num_points,num_points))
    conditions.append(zero_tensor)

def my_function(i, j, k, conditions):
    print((i,j,k))
    x=np.array([[all_x[i,j,k]],[all_y[i,j,k]],[all_z[i,j,k]]]);
    tmp,_=get_conditions(x)
    return tmp


result=Parallel(n_jobs=40)(delayed(my_function)(i, j, k, conditions) for i in range(all_x.shape[0]) for j in range(all_x.shape[1]) for k in range(all_x.shape[2]))

print(len(result))
print(len(result[0]))

index=0;
for i in range(all_x.shape[0]):
    for j in range(all_x.shape[1]):
        for k in range(all_x.shape[2]):

            for index_cond in range(num_of_conditions):#For each condition
                conditions[index_cond][i,j,k]=result[index][index_cond]

            index+=1


conditions_processed=[]
for i in range(len(conditions)):

    cond_i=copy.deepcopy(conditions[i])
    for j in range(len(conditions)):
        if (i==j):
            continue
        cond_j=conditions[j]
        
        cond_j_is_ineq=all_is_ineq_condition[j]

        if(cond_j_is_ineq):
            cond_i[cond_j>0.0]=None #See https://stackoverflow.com/questions/40461045/mayavi-combining-two-implicit-3d-surfaces
        else:
            cond_i[cond_j>0.0]=None
            cond_i[cond_j<0.0]=None

    conditions_processed.append(cond_i)


for i in range(len(conditions_processed)):
    # print(i)
    cond_i=conditions_processed[i]
    # print(cond_i)
    if(np.isnan(cond_i).all()):
        continue
    maximum=np.nanmax(cond_i)
    minimum=np.nanmin(cond_i)
    if(maximum<0 or minimum>0):
        print("Continue")
        continue;
    color=tuple(random.random() for _ in range(3))
    tmp=mlab.contour3d(all_x,all_y,all_z,cond_i, contours = [0], color=color, opacity=1.0) 
    # tmp.actor.property.interpolation = 'phong' #https://stackoverflow.com/a/31754643
    # tmp.actor.property.specular = 0.9
    # tmp.actor.property.specular_power = 128

lensoffset=0.0
xx = yy = zz = np.arange(0.0,1.5,0.1)
xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
mlab.plot3d(yx,yy+lensoffset,yz,line_width=0.01,tube_radius=0.02)
mlab.plot3d(zx,zy+lensoffset,zz,line_width=0.01,tube_radius=0.02)
mlab.plot3d(xx,xy+lensoffset,xz,line_width=0.01,tube_radius=0.02)

print("Showing!")
mlab.show()
