import numpy as np
import utils

def getCube():
	A1=np.array([ [1, 0, 0],
				 [0, 1, 0],
				 [0, 0, 1],
				 [-1, 0, 0],
				 [0, -1, 0],
				 [0, 0, -1]]);

	b1=np.array([[1],
				[1],
				[1],
				[0],
				[0],
				[0]])

	return A1, b1

#Ellipsoid is defined as {x | (x-c)'E(x-c)<=1}
#Where E is a positive semidefinite matrix
def getEllipsoid(E, c):
	#Convert to (1/2)x'P_ix + q_i'x +r_i <=0
	P=2*E;
	q=(-2*E@c)
	r=c.T@E@c-1
	return P, q, r

#Sphere of radius r centered around c
def getSphere(r, c):
	return getEllipsoid((r/2.0)*np.eye(c.shape[0]),c)

def getParaboloid3D():
	P=np.array([[1.0, 0.0, 0.0],
				[0.0, 1.0, 0.0],
				[0.0, 0.0, 0.0]])
	q=np.array([[0.0],[0.0],[-1.0]])
	r=np.array([[0.0]])

	return P,q,r

# def convertEachItemToList():


def getNoneLinearConstraints():
	return None, None, None, None

def getNoneQuadraticConstraints():
	return None, None, None

def getExample(example):

	A1, b1, A2, b2 = getNoneLinearConstraints()
	all_P, all_q, all_r = getNoneQuadraticConstraints()


	if example==0: #A polygon embeded in 3D
		A1, b1=getCube()
		A2=np.array([[1, 1, 1]]);
		b2=np.array([[1]]);

	elif example==1: #A polygon embeded in 3D with an sphere

		A1, b1=getCube()
		A2=np.array([[1, 1, 1]]);
		b2=np.array([[1]]);

		P,q,r=getSphere(2.0,np.zeros((3,1)))

		all_P=[P]
		all_q=[q]
		all_r=[r]

	elif example==2: #Just a sphere

		P,q,r=getSphere(2.0,np.zeros((3,1)))

		all_P=[P]
		all_q=[q]
		all_r=[r]

	elif example==3: #Just a paraboloid

		P,q,r=getParaboloid3D()

		all_P=[P]
		all_q=[q]
		all_r=[r]

	#A 2d polyhedron 
	elif (example==4  
	#A 2d polyhedron with a cirle
	     or example==5):   
		A1=np.array([[-1,0],
					 [0, -1],
					 [0, 1],
					 [0.2425,    0.9701]]);

		b1=np.array([[0],
					[0],
					[1],
					[1.2127]])

		if(example==5):
			P,q,r=getSphere(1.0,np.zeros((2,1)))
			all_P=[P]
			all_q=[q]
			all_r=[r]



	elif example==6: #The intersection between a cube and a plane 3d cube 
		A1, b1=getCube()
		A2=np.array([[1, 1, 1],
					  [-1, 1, 1] ]);
		b2=np.array([[1],[0.1]]);

	elif example==7: #Just a plane
		A2=np.array([[1, 1, 1]]);
		b2=np.array([[1]]);	


	elif example==8: #Unbounded 2d polyhedron. It has two vertices and two rays

		A1=np.array([[0,-1], [2,-4], [-2,1]]);
		b1=np.array([[-2], [8], [-5]]);

	else:
		raise Exception("Not implemented yet")

	return utils.linearAndConvexQuadraticConstraints(A1, b1, A2, b2, all_P, all_q, all_r)