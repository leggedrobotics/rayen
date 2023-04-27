import torch
import numpy as np

# u = torch.ones((3,1), requires_grad=True)
 
# f = u.T@u
 
# # print(u)
# # print(v)
# print(f)
 
# f.backward()
# print("Partial derivative with respect to u: \n", u.grad)
# # print("Partial derivative with respect to v: ", v.grad)


# import torch

# a = torch.tensor([2., 3.], requires_grad=True)
# b = torch.tensor([6., 4.], requires_grad=True)

# Q = 3*a**3 - b**2

# print(Q.shape)

# external_grad = torch.tensor([1., 1.])
# Q.backward(gradient=external_grad)

# # check if collected gradients are correct
# print(9*a**2 == a.grad)
# print(-2*b == b.grad)


# def exp_reducer(x):
#     return x.exp().sum(dim=1)

# inputs = torch.rand(2, 2)
# result=torch.autograd.functional.jacobian(exp_reducer, inputs)


P=np.array([[1,4, 7],[5,6,8]])

print(f"P is \n{P}")

print("------------------")
print(P[:,-1])
print(P[0:-1,[-1]])

print(P[list(range(P.shape[0])),[-1]])