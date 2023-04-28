

from root import root_find
import torch
import cvxpy as cp

# initial guess


>>> x.value = 3
>>> Si[0].value

x_cp = cp.Variable(2,1)
expression=x_cp[0,0]*x_cp[0,0] - x_cp[1,0]


# function to optimize
def my_func(x):
    x_cp.value=x;
    expression=
    return x[0,0]*x[0,0] - x[1,0] #Set is x^2-y<=0
    # return x - torch.cos(x)

def my_tracer(alpha):
    v_n=torch.Tensor([[0.5],[1]])
    z0=torch.Tensor([[0],[0.1]])
    return my_func(z0+alpha*alpha*v_n)

def newton(func, guess, runs=5000): 
    for _ in range(runs): 
        # evaluate our function with current value of `guess`
        value = func(guess)
        value.backward()
        # update our `guess` based on the gradient
        guess.data -= (value / guess.grad).data
        # zero out current gradient to hold new gradients in next iteration 
        guess.grad.data.zero_() 
        print(guess)
    return guess.data # return our final `guess` after 5 updates

# call starts
guess = torch.Tensor([1], dtype=torch.float64, requires_grad = True) 
result = newton(my_tracer, guess)

print(my_tracer(result))
# output of `result`
# tensor([0.7391], dtype=torch.float64)


# Taken from https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/numerics/solvers/root.py

# import pytest
# from torchdyn.numerics.solvers.root import *
# from .conftest import *

# def func(x):
#     return x+5

# z0 = -4.5 * torch.randn(1, 1)
# search_method='armijo';
# method='broyden_fast'        # 'broyden', 'broyden_fast' 'newton'

# result = root_find(func, z0, alpha=.4, search_method=search_method, f_tol=1e-3, f_rtol=1e-3, x_tol=1e-4, x_rtol=1e-2,  maxiters=5000, method=method, verbose=False)

# print(func(result))