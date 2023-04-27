import numpy as np
import constraints


def getCube():
	A1=np.array([ [1.0, 0, 0],
				 [0, 1.0, 0],
				 [0, 0, 1.0],
				 [-1.0, 0, 0],
				 [0, -1.0, 0],
				 [0, 0, -1.0]]);

	b1=np.array([[1.0],
				[1.0],
				[1.0],
				[0],
				[0],
				[0]])

	return A1, b1

#Ellipsoid is defined as {x | (x-c)'E(x-c)<=1}
#Where E is a positive semidefinite matrix
def getEllipsoidConstraint(E, c):
	#Convert to (1/2)x'P_ix + q_i'x +r_i <=0
	P=2*E;
	q=(-2*E@c)
	r=c.T@E@c-1
	return constraints.convexQuadraticConstraint(P, q, r)

#Sphere of radius r centered around c
def getSphereConstraint(r, c):
	return getEllipsoidConstraint((1/(r*r))*np.eye(c.shape[0]),c)

def getParaboloid3DConstraint():
	P=np.array([[1.0, 0.0, 0.0],
				[0.0, 1.0, 0.0],
				[0.0, 0.0, 0.0]])
	q=np.array([[0.0],[0.0],[-1.0]])
	r=np.array([[0.0]])

	return constraints.convexQuadraticConstraint(P,q,r)

def getSOC3DConstraint():
	M=np.array([[1.0, 0.0, 0.0],
				[0.0, 1.0, 0.0],
				[0.0, 0.0, 0.0]])
	s=np.array([[0.0],[0.0],[0.0]])
	c=np.array([[0.0],[0.0],[1.0]])
	d=np.array([[0.0]])

	return constraints.SOCConstraint(M, s, c, d)

def getPSDCone3DConstraint():
	#[x y;y z] >> 0
	F0=np.array([[1.0, 0.0],
				 [0.0, 0.0]])

	F1=np.array([[0.0, 1.0],
				 [1.0, 0.0]])

	F2=np.array([[0.0, 0.0],
				 [0.0, 1.0]])

	F3=np.array([[0.0, 0.0],
				 [0.0, 0.0]])

	return constraints.SDPConstraint([F0, F1, F2, F3])

def getNoneLinearConstraints():
	return None, None, None, None

def getNoneQuadraticConstraints():
	return [], [], []

def getExample(example):

	# A1, b1, A2, b2 = getNoneLinearConstraints()
	# all_P, all_q, all_r = getNoneQuadraticConstraints()
	lc=None
	qcs=[]
	socs=[]
	sdpc=None

	if example==0: #A 2D polygon embeded in 3D
		A1, b1=getCube()
		A2=np.array([[1.0, 1.0, 1.0]]);
		b2=np.array([[1.0]]);
		lc=constraints.LinearConstraint(A1, b1, A2, b2)

	elif example==1: #A polygon embeded in 3D with an sphere

		A1, b1=getCube()
		A2=np.array([[1.0, 1.0, 1.0]]);
		b2=np.array([[1.0]]);
		lc=constraints.LinearConstraint(A1, b1, A2, b2)
		qcs.append(getSphereConstraint(0.8,np.zeros((3,1))))


	elif example==2: #Just a sphere

		qcs.append(getSphereConstraint(2.0,np.zeros((3,1))))

	elif example==3: #Just a paraboloid

		qcs.append(getParaboloid3DConstraint())

	#A 2d polyhedron 
	elif (example==4  
	#A 2d polyhedron with a cirle
	     or example==5):   
		A1=np.array([[-1,0],
					 [0, -1.0],
					 [0, 1.0],
					 [0.6,    0.9701]]);

		b1=np.array([[0],
					[0],
					[1],
					[1.2127]])

		lc=constraints.LinearConstraint(A1, b1, None, None)

		if(example==5):
			qcs.append(getSphereConstraint(1.25,np.zeros((2,1))))

	elif example==6: #The intersection between a cube and two planes 
		A1, b1=getCube()
		A2=np.array([[1.0, 1.0, 1.0],
					  [-1.0, 1.0, 1.0] ]);
		b2=np.array([[1.0],[0.1]]);
		lc=constraints.LinearConstraint(A1, b1, A2, b2)

	elif example==7: #Just a plane
		A2=np.array([[1.0, 1.0, 1.0]]);
		b2=np.array([[1.0]]);	
		lc=constraints.LinearConstraint(None, None, A2, b2)


	elif example==8: #Unbounded 2d polyhedron. It has two vertices and two rays

		A1=np.array([[0.0,-1.0], [2.0,-4.0], [-2.0,1.0]]);
		b1=np.array([[-2.0], [1.0], [-5.0]]);
		lc=constraints.LinearConstraint(A1, b1, None, None)

	elif example==9: #A paraboloid and a plane
		qcs.append(getParaboloid3DConstraint())

		A2=np.array([[1.0, 1.0, 3.0]]);
		b2=np.array([[1.0]]);	
		lc=constraints.LinearConstraint(None, None, A2, b2)	

	elif example==10: #A paraboloid and a shpere
		qcs.append(getParaboloid3DConstraint())
		qcs.append(getSphereConstraint(2.0,np.zeros((3,1))))	

	elif example==11: #A second-order cone 
		socs.append(getSOC3DConstraint())

	elif example==12: #The PSD cone in 3D
		sdpc = getPSDCone3DConstraint()

	elif example==13: #Many of them
		A1=np.array([[-1.0,-1.0,-1.0]])
		b1=np.array([[-1.0]])
		lc=constraints.LinearConstraint(A1, b1, None, None)
		E_ellipsoid=np.array([[0.1,0,0],
							  [0.0,1.0,0.0],
							  [0.0,0.0,1.0]])
		qcs.append(getEllipsoidConstraint(E_ellipsoid, np.zeros((3,1))))
		socs.append(getSOC3DConstraint())
		sdpc = getPSDCone3DConstraint()

	elif example==14: #Many of them
		A1=np.array([[-1.0,-1.0,-1.0],
			         [-1.0,2.0,2.0]])
		b1=np.array([[-1.0],[1.0]])
		lc=constraints.LinearConstraint(A1, b1, None, None)
		E_ellipsoid=np.array([[0.6,0,0],
							  [0.0,1.0,0.0],
							  [0.0,0.0,1.0]])
		qcs.append(getEllipsoidConstraint(E_ellipsoid, np.zeros((3,1))))
		# qcs.append(getParaboloid3DConstraint())
		# socs.append(getSOC3DConstraint())
		# sdpc = getPSDCone3DConstraint()

	else:
		raise Exception("Not implemented yet")


	return constraints.convexConstraints(lc=lc, qcs=qcs, socs=socs, sdpc=sdpc)