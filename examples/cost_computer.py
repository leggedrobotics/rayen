import torch
import torch.nn as nn
import numpy as np
import scipy
import cvxpy as cp
import math
from cvxpylayers.torch import CvxpyLayer
import random
import copy

import fixpath #Following this example: https://github.com/tartley/colorama/blob/master/demos/demo01.py
from rayen import utils



class CostComputer(nn.Module): #Using nn.Module to be able to use register_buffer (and hence to be able to have the to() method)
	def __init__(self, cs):
		super().__init__()

		if(cs.has_quadratic_constraints):
			all_P, all_q, all_r = utils.getAllPqrFromQcs(cs.qcs)
			self.register_buffer("all_P", torch.Tensor(np.array(all_P)))
			self.register_buffer("all_q", torch.Tensor(np.array(all_q)))
			self.register_buffer("all_r", torch.Tensor(np.array(all_r)))

		if(cs.has_soc_constraints):
			all_M, all_s, all_c, all_d = utils.getAllMscdFromQcs(cs.socs)
			self.register_buffer("all_M", torch.Tensor(np.array(all_M)))
			self.register_buffer("all_s", torch.Tensor(np.array(all_s)))
			self.register_buffer("all_c", torch.Tensor(np.array(all_c)))
			self.register_buffer("all_d", torch.Tensor(np.array(all_d)))

		#See https://discuss.pytorch.org/t/model-cuda-does-not-convert-all-variables-to-cuda/114733/9
		# and https://discuss.pytorch.org/t/keeping-constant-value-in-module-on-correct-device/10129
		self.register_buffer("A_p", torch.Tensor(cs.A_p))
		self.register_buffer("b_p", torch.Tensor(cs.b_p))
		self.register_buffer("yp", torch.Tensor(cs.yp))
		self.register_buffer("NA_E", torch.Tensor(cs.NA_E))
		self.register_buffer("z0", torch.Tensor(cs.z0))

		self.has_linear_ineq_constraints=cs.has_linear_ineq_constraints
		self.has_linear_eq_constraints=cs.has_linear_eq_constraints
		self.has_quadratic_constraints=cs.has_quadratic_constraints
		self.has_soc_constraints=cs.has_soc_constraints
		self.has_lmi_constraints=cs.has_lmi_constraints

		if self.has_linear_ineq_constraints:
			self.register_buffer("A1", torch.Tensor(cs.lc.A1))
			self.register_buffer("b1", torch.Tensor(cs.lc.b1))

		if self.has_linear_eq_constraints:
			self.register_buffer("A2", torch.Tensor(cs.lc.A2))
			self.register_buffer("b2", torch.Tensor(cs.lc.b2))

	def getyFromz(self, z):
		y=self.NA_E@z + self.yp
		return y

	#This function below assumes that y is in the plane spanned by the columns of NA_E!!
	# def getzFromy(self, y):
	# 	z=self.NA_E.T@(y - self.yp)
	# 	return z

	def getSumSoftCostAllSamples(self, y):
		################## STACK THE VALUES OF ALL THE INEQUALITIES
		all_inequalities=torch.empty((y.shape[0],0,1), device=y.device)
		##### Ap*z<=bp
		# z=self.getzFromy(y); #I cannot use this function, since this function assumes that y lies in \mathcal{Y}_L
							   #I could use it for DC3, but not for the "UU method"
		# all_inequalities=torch.cat((all_inequalities, self.A_p@z-self.b_p), dim=1);
		##### A1*y<=b1
		if self.has_linear_ineq_constraints:
			all_inequalities=torch.cat((all_inequalities, self.A1@y-self.b1), dim=1);

		##### g(y)<=0
		if(self.has_quadratic_constraints):
			for i in range(self.all_P.shape[0]):
				P=self.all_P[i,:,:]
				q=self.all_q[i,:,:]
				r=self.all_r[i,:,:]
				all_inequalities=torch.cat((all_inequalities, utils.quadExpression(y=y,P=P,q=q,r=r)), dim=1)

		if(self.has_soc_constraints):
			for i in range(self.all_M.shape[0]):
				M=self.all_M[i,:,:]
				s=self.all_s[i,:,:]
				c=self.all_c[i,:,:]
				d=self.all_d[i,:,:]

				lhs=torch.linalg.vector_norm(M@y + s, dim=1, keepdim=True) - c.T@y - d

				all_inequalities=torch.cat((all_inequalities, lhs), dim=1)

		if(self.has_lmi_constraints):
			raise NotImplementedError 

		soft_cost = torch.sum(torch.square(torch.nn.functional.relu(all_inequalities)))

		if self.has_linear_eq_constraints:
			soft_cost_equalities = torch.sum(torch.square(self.A2@y-self.b2)); #This is zero for the DC3 method, and nonzero for the UU method
			soft_cost += soft_cost_equalities

		########################################################################

		return soft_cost


	def getSumObjCostAllSamples(self, y, Pobj, qobj, robj):
		tmp=utils.quadExpression(y=y,P=Pobj,q=qobj,r=robj)
		assert tmp.shape==(y.shape[0], 1, 1)

		return torch.sum(tmp)

	def getSumSupervisedCostAllSamples(self, y, y_predicted):
		return torch.sum(torch.square(y-y_predicted))

	def getSumLossAllSamples(self, params, y, y_predicted, Pobj, qobj, robj, isTesting=False):

		loss=0.0;

		if(params['use_supervised']):
			loss += self.getSumSupervisedCostAllSamples(y, y_predicted)
		else:
			loss += self.getSumObjCostAllSamples(y_predicted, Pobj, qobj, robj)

		if (isTesting==False and params['weight_soft_cost']>0): #I need the term params['weight_soft_cost']>0 because the soft cost is not implemented for LMI constraints yet
			loss += params['weight_soft_cost']*self.getSumSoftCostAllSamples(y_predicted)

		# loss=params['use_supervised']*self.getSumSupervisedCostAllSamples(y, y_predicted) + \
		# 	 (1-isTesting)*params['weight_soft_cost']*self.getSumSoftCostAllSamples(y_predicted) + \
		# 	 (1-params['use_supervised'])*self.getSumObjCostAllSamples(y_predicted, Pobj, qobj, robj)

		return loss
