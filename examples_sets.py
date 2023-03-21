import numpy as np
from utils import LinearConstraint

def getCube():
	Aineq=np.array([ [1, 0, 0],
				 [0, 1, 0],
				 [0, 0, 1],
				 [-1, 0, 0],
				 [0, -1, 0],
				 [0, 0, -1]]);

	bineq=np.array([[1],
				[1],
				[1],
				[0],
				[0],
				[0]])

	return Aineq, bineq


def getExample(example):

	if example==0: #A polygon embeded in 3D

		Aineq, bineq=getCube()
		Aeq=np.array([[1, 1, 1]]);
		beq=np.array([[1]]);

	elif example==1: #A polyhedron
		Aineq=np.array([[-1,0],
					 [0, -1],
					 [0, 1],
					 [0.2425,    0.9701]]);

		bineq=np.array([[0],
					[0],
					[1],
					[1.2127]])

		Aeq=None;
		beq=None;

	elif example==2: #The intersection between a cube and a plane 3d cube 
		Aineq, bineq=getCube()
		Aeq=np.array([[1, 1, 1],
					  [-1, 1, 1] ]);
		beq=np.array([[1],[0.1]]);

	elif example==3: #Just a plane
		Aineq=None
		bineq=None
		Aeq=np.array([[1, 1, 1]]);
		beq=np.array([[1]]);	

	elif example==4: #Unbounded 2d polyhedron. It has two vertices and two rays

		Aineq=np.array([[0,-1], [2,-4], [-2,1]]);
		bineq=np.array([[-2], [8], [-5]]);
		Aeq=None
		beq=None

	else:
		raise Exception("Not implemented yet")

	return LinearConstraint(Aineq, bineq, Aeq, beq)