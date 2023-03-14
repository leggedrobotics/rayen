import numpy as np
import torch
import torch.nn as nn
from linear_constraint_barycentric import LinearConstraintBarycentric
from linear_constraint_layer import LinearConstraintLayer
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

class WrapperWalkerForImages(nn.Module):
    def __init__(self, A_np, b_np):
        super().__init__()
        dim = A_np.shape[1]

        if(b_np.ndim==1):
            b_np=np.expand_dims(b_np, 1)

        self.constraint = LinearConstraintWalker(A_np,b_np, None, None)

        mapper=nn.Sequential(nn.Linear(dim, self.constraint.getNumelInputWalker()))
        self.constraint.setMapper(mapper)

    def forward(self, x):
        orig_size = x.size()
        y = x.view(x.size(0), -1)
        z = self.constraint(y)
        z = z.view(orig_size)
        return z

