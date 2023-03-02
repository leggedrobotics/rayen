import sys
import os
import time
import numpy as np
import numpy.linalg as la
import torch
import torch.nn as nn
import torch.nn.functional as F
import cdd
from osqp_projection import ConstraintProjector

def H_to_V(A, b):
    """
    Converts a polyhedron in H-representation to
    one in V-representation using pycddlib.
    """
    # define cdd problem and convert representation
    if len(b.shape) == 1:
        b = np.expand_dims(b, axis=1)
    mat_np = np.concatenate([b, -A], axis=1)
    if mat_np.dtype in [np.int32, np.int64]:
        nt = 'fraction'
    else:
        nt = 'float'
    mat_list = mat_np.tolist()

    mat_cdd = cdd.Matrix(mat_list, number_type=nt)
    mat_cdd.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat_cdd)
    gen = poly.get_generators()

    # convert the cddlib output data structure to numpy
    V_list = []
    R_list = []
    lin_set = gen.lin_set
    V_lin_idx = []
    R_lin_idx = []
    for i in range(gen.row_size):
        g = gen[i]
        g_type = g[0]
        g_vec = g[1:]
        if i in lin_set:
            is_linear = True
        else:
            is_linear = False
        if g_type == 1:
            V_list.append(g_vec)
            if is_linear:
                V_lin_idx.append(len(V_list) - 1)
        elif g_type == 0:
            R_list.append(g_vec)
            if is_linear:
                R_lin_idx.append(len(R_list) - 1)
        else:
            raise ValueError('Generator data structure is not valid.')

    V = np.asarray(V_list)
    R = np.asarray(R_list)

    # by convention of cddlib, those rays assciated with R_lin_idx
    # are not constrained to non-negative coefficients
    if len(R) > 0:
        R = np.concatenate([R, -R[R_lin_idx, :]], axis=0)
    return V, R

class LinearConstraint(nn.Module):
    """
    A linear inequality constraint layer.

    A_np, b_np are numpy ndarrays that define the constraint set A_np.dot(x) <= b_np,
    which are enforced at training time.

    box_constraints is a tuple of (min_val, max_val), which are additionally enforced
    at test time.

    If filename is provided, then the model either reads an existing constraint in
    V-representation or first computes and then saves this representation.
    In the first case, the arguments A_np, b_np are ignored.
    All constraints here are cones, possibly lifted by one dimension.

    """
    def __init__(self, A_np, b_np, box_constraints=False):
        super().__init__()
        self.box_constraints = box_constraints
        if box_constraints:
            print('Enforcing box constraints.')

        print('Computing constraints in V-representation.')
        if np.allclose(b_np, 0):
            print("Constraint set is a zero-centered cone.")
        else:
            raise ValueError("Constraint set is NOT a zero-centered cone.")
        A_dd = A_np
        b_dd = np.zeros_like(b_np)
        num_constraints = A_np.shape[0]
        _, R_dd = H_to_V(A_dd, b_dd)
        R_dd = np.float32(R_dd)
        A_np = np.float32(A_np)
        b_np = np.float32(b_np)

        self.A_th = torch.FloatTensor(A_np)
        self.b_th = torch.FloatTensor(b_np)

        # control the scale of the rays and convert to torch.Tensor
        R_dd = R_dd / la.norm(R_dd, axis=1, keepdims=True)
        dim = R_dd.shape[1]
        self.R = torch.FloatTensor(R_dd)

        # define the linear layer that maps to the parameterization coefficients
        self.beta_lin = nn.Linear(self.R.size(1), self.R.size(0))
        nn.init.uniform_(self.beta_lin.weight, a=-1/dim, b=1/dim)
        nn.init.constant_(self.beta_lin.bias, 0.)
        self.bn = nn.BatchNorm1d(A_dd.shape[1])

        self.scaling_epoch = 25

    def check_device(self):
        # make sure the ray tensor and the model live on the same device
        module_device = self.beta_lin.weight.device
        self.R = self.R.to(module_device)

    def forward(self, x, epoch=-1):
        self.check_device()

        orig_size = x.size()
        x = x.view(x.size(0), -1)

        # map this input to coefficients beta of the V-parameterization
        # use absolute value to enforce non-negativity of the conical coefficients
        x = self.bn(x)
        beta = self.beta_lin(x)
        beta_abs = torch.abs(beta)
        x = beta_abs.mm(self.R)

        # divide by infinity norm if that is larger than one
        # this ensures a [-1,1] box constraint and still obeys the conic constraints due to pure scaling
        if self.box_constraints and (not self.training or ((epoch > self.scaling_epoch) and (epoch > 0))):
            box_scaling = torch.clamp(torch.max(torch.abs(x), dim=1, keepdim=True)[0], min=1)
            x = x / box_scaling

        return x.view(orig_size)
