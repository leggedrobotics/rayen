# --------------------------------------------------------------------------
# Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich
# See LICENSE file for the license information
# --------------------------------------------------------------------------

# Measures the preprocessing (offline phase, Section 3 of the paper) times of Optimizations 1 and 2 (corridor_dim2.mat and corridor_dim3.mat)
# The time of each step of the offline phase is obtained by timing all the calls to cvxpy.Problem.solve(), and assigning each call
# to the step of ConvexConstraints.__init__() it comes from.

import time
import inspect
import numpy as np
import torch
import cvxpy as cp
import pandas as pd

from create_dataset import getCorridorConstraints
import fixpath #Following this example: https://github.com/tartley/colorama/blob/master/demos/demo01.py
from rayen import constraints, constraint_module

torch.set_default_dtype(torch.float64)

num_repetitions=1

utils_source, first_line=inspect.getsourcelines(constraints.ConvexConstraints.__init__)
def getLineOf(text):
	return first_line + next(i for i, line in enumerate(utils_source) if text in line)

#Steps of the offline phase, delimited by the comments in ConvexConstraints.__init__()
steps=[('Feasibility check',                getLineOf('Ensure that the feasible set is not empty')),
       ('Removal of redundant constraints', getLineOf('#Remove redundant constraints')),
       ('Affine hull (equality set)',       getLineOf('#Find equality set')),
       ('Nullspace projection',             getLineOf('#Project into the nullspace of A_E')),
       ('Interior point',                   getLineOf('#Obtain a strictly feasible point z0')),
       ('Setup of projection problem',      getLineOf('SET UP PROBLEM FOR PROJECTION'))]

def getStep(lineno):
	return [name for (name, start) in steps if start<=lineno][-1]

stats={}
original_solve=cp.Problem.solve
def timedSolve(self, *args, **kwargs):
	caller_line=inspect.stack()[1].lineno
	start=time.perf_counter()
	result=original_solve(self, *args, **kwargs)
	elapsed=time.perf_counter()-start
	step=getStep(caller_line)
	stats[step]['num_solves']+=1
	stats[step]['time_solve_s']+=elapsed
	stats[step]['time_solver_s']+=self.solver_stats.solve_time
	return result
cp.Problem.solve=timedSolve

#Warm up Gurobi (license check, etc.)
x=cp.Variable(); original_solve(cp.Problem(cp.Minimize(x), [x>=0]), solver=cp.GUROBI)

all_results=[]
for dimension, name in [(2, 'Optimization 1'), (3, 'Optimization 2')]:
	for repetition in range(num_repetitions):
		stats.clear()
		for (step_name, _) in steps:
			stats[step_name]={'num_solves': 0, 'time_solve_s': 0.0, 'time_solver_s': 0.0}

		start=time.perf_counter()
		cs=getCorridorConstraints(dimension)
		time_total_s=time.perf_counter()-start

		start=time.perf_counter()
		layer=constraint_module.ConstraintModule(cs, input_dim=64, method='RAYEN', create_map=True)
		time_layer_s=time.perf_counter()-start

		assert cs.solver=='GUROBI'
		for step_name in ['Feasibility check', 'Removal of redundant constraints', 'Affine hull (equality set)', 'Interior point']:
			all_results.append({'problem': name, 'repetition': repetition, 'step': step_name, **stats[step_name]})
		all_results.append({'problem': name, 'repetition': repetition, 'step': 'Rest (loading, nullspace, model building,...)', 'num_solves': 0, 'time_solve_s': time_total_s-sum(s['time_solve_s'] for s in stats.values()), 'time_solver_s': 0.0})
		all_results.append({'problem': name, 'repetition': repetition, 'step': 'Total offline phase', 'num_solves': sum(s['num_solves'] for s in stats.values()), 'time_solve_s': time_total_s, 'time_solver_s': sum(s['time_solver_s'] for s in stats.values())})
		all_results.append({'problem': name, 'repetition': repetition, 'step': 'Computation of D, phi, delta (ConstraintModule)', 'num_solves': 0, 'time_solve_s': time_layer_s, 'time_solver_s': 0.0})

	print(f"{name}: k={cs.k}, rows of A1={cs.lc.A1.shape[0]}, rows of A2={cs.lc.A2.shape[0]}, eta={len(cs.qcs)}, n={cs.n}, rows of A_p={cs.A_p.shape[0]}")

df=pd.DataFrame(all_results)
df=df.groupby(['problem', 'step'], sort=False).agg(num_solves=('num_solves', 'mean'), time_s_mean=('time_solve_s', 'mean'), time_s_std=('time_solve_s', 'std'), time_gurobi_s_mean=('time_solver_s', 'mean'))
pd.set_option('display.width', 200)
print(df)
df.to_csv('./scripts/results/preprocessing_time.csv')
