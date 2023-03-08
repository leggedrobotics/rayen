from linear_constraint_walker import Preprocessor
import numpy as np


A=np.array([ [1, 0, 0],
			 [0, 1, 0],
			 [0, 0, 1],
			 [-1, 0, 0],
			 [0, -1, 0],
			 [0, 0, -1]]);

b=np.array([[1],
			[1],
			[1],
			[0],
			[0],
			[0]])


Aeq=np.array([[1, 0, 0]]);
beq=np.array([[0]]);


p=Preprocessor(A,b,Aeq,beq)