import osqp
import numpy as np
import scipy.sparse as sparse
import scipy.linalg as sla
import warnings

OSQP_SOLVED = 1

class ConstraintProjector(object):
    def __init__(self, A, b, box_constraints=None, polish=True):
        m, d = A.shape
        if not sparse.issparse(A):
            A = sparse.csc_matrix(A)

        if box_constraints is not None:
            min_box, max_box = box_constraints
            I = sparse.eye(d)
            A = sparse.vstack([A,I])
            l = np.concatenate([np.NINF * np.ones(m), min_box * np.ones(d)], axis=0)
            u = np.concatenate([b, max_box * np.ones(d)], axis=0)
        else:
            l = np.NINF * np.ones(m)
            u = b
        
        self.A = A
        self.l = l
        self.u = u

        # define QP problem dummy matrices which are not needed for the projection
        P = sparse.eye(d)
        q = np.zeros(d)
        self.d = d

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(P, q, A, l, u,
                        eps_abs=1e-6,
                        eps_rel=1e-6,
                        eps_prim_inf=1e-6,
                        eps_dual_inf=1e-6,
                        verbose=False,
                        polish=polish,
                        warm_start=True)

    def is_feasible_point(self, x, eps=1e-4):
        return np.all(self.A.dot(x) + eps > self.l) and np.all(self.A.dot(x) - eps < self.u)

    def project(self, x):
        try:
            self.prob.update(q = -x)
            self.prob.warm_start(x=x)
        except:
            print('Size of q is {} and size of new x is {}.'.format(self.d, x.shape))
        res = self.prob.solve()
        if res.info.status_val != OSQP_SOLVED:
            warnings.warn('Problem not solved by OSQP. Solver status is: {}'.format(res.info.status), RuntimeWarning)
        argmin = res.x
        min_value = 2*res.info.obj_val + x.dot(x)
        return min_value, argmin


class BatchProjector(object):
    def __init__(self, A, b, box_constraints=None):
        self.projector = ConstraintProjector(A, b, box_constraints=box_constraints)

    def project(self, data):
        argmins = np.zeros_like(data)
        min_values = np.zeros(data.shape[0])
        for k, x in enumerate(data):
            x_flat = x.flatten()
            min_value, argmin = self.projector.project(x_flat)
            min_values[k] = min_value
            argmins[k] = argmin.reshape(x.shape)
        return np.mean(min_values), argmins
