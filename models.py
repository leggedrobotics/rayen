import numpy as np
import torch
import torch.nn as nn
from linear_constraint_barycentric import LinearConstraintBarycentric
from linear_constraint_walker import LinearConstraintWalker
from osqp_projection import ConstraintProjector


class ProjectionModel(nn.Module):
    def __init__(self, A_np, b_np, mapping=None, box_constraints=None):
        super().__init__()
        d = A_np.shape[1]
        self.net = nn.Sequential() if mapping is None else mapping
        self.A_np = A_np
        self.b_np = b_np
        self.projector = ConstraintProjector(A_np, b_np, box_constraints=box_constraints)

    def forward(self, x, epoch=-1):
        orig_size = x.size()
        x = x.view(x.size(0), -1)
        x = self.net(x)
        if not self.training:
            # compute projected solution
            out = torch.zeros_like(x)
            for k, x_k in enumerate(x):
                _, projection_argmin = self.projector.project(x_k.detach().cpu().numpy().flatten())
                out[k] = torch.FloatTensor(projection_argmin).type_as(x)
            x = out
        return x.view(orig_size)


class BarycentricModel(nn.Module):
    def __init__(self, A_np, b_np, mapping=None, box_constraints=None, filename=''):
        super().__init__()
        d = A_np.shape[1]
        self.net = nn.Sequential() if mapping is None else mapping
        self.constraint = LinearConstraintBarycentric(A_np, b_np, box_constraints)

    def forward(self, x, epoch=-1):
        orig_size = x.size()
        y = x.view(x.size(0), -1)
        z = self.constraint(self.net(y), epoch)
        z = z.view(orig_size)
        return z

class MapperAndWalker(nn.Module):
    def __init__(self, A_np, b_np, num_steps, numel_input_mapper):
        super().__init__()
        self.constraint = LinearConstraintWalker(A_np, b_np, num_steps=num_steps, use_max_ellipsoid=False)
        print(f"numel_input_mapper={numel_input_mapper}")
        print(f"self.constraint.getNumelInput()={self.constraint.getNumelInput()}")
        self.net = nn.Sequential(nn.Linear(numel_input_mapper, self.constraint.getNumelInput()))

    def forward(self, x):
        orig_size = x.size()
        y = x.view(x.size(0), -1)
        z = self.constraint(self.net(y))
        z = z.view(orig_size)
        return z

