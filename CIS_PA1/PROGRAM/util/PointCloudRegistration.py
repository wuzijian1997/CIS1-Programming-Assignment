import numpy as np
from util.Cartesian_transformation import Cartesian_transformation

"""
    In this file, we develop a module for point-cloud-to-point-cloud registration. We implement
    the Point2Point_Reg class with two point sets as input and return a Cartesin_transformation F.
"""

def Direct_solveRotation(points_a, points_b, epsilon=1e-6):
    H = np.matmul(points_a, np.transpose(points_b))
    assert H.shape[0] == H.shape[1] and H.shape[0] == 3
    u, s, vh = np.linalg.svd(H, full_matrices=False)
    R = np.matmul(u, vh) 
    R = np.transpose(R)
    det = np.linalg.det(R)
    if abs(det -1) < epsilon:
        return R
    elif abs(det +1) < epsilon:
        s[s<epsilon]=0
        if 0 in s:
            u[:,-1] *= -1
            return np.transpose(np.matmul(u, vh))
        else:
            #return Iterative_solveRotation(points_a, points_b) 
            # find outlier and remove
            return np.zeros_like(R)
    
def Direct_QuartSolveRotaion(points_a, points_b, epsilon=1e-6):

    return NotImplemented

def Iterative_solveRotation(points_a, points_b, epsilon=1e-6):
    error = lambda A, B, R:\
        np.matmul(\
            np.concatenate(\
                [np.concatenate([R, -B[:,i][:,None]], axis = 1) for i in range(B.shape[1])], axis=1),
            np.concatenate([A,np.ones((1,A.shape[1]))], axis=0).reshape(4*A.shape[1], 1))      
    # initial guess of R 
    R_0 = np.matmul(np.matmul(points_b, np.transpose(points_b)), np.linalg.inv(np.matmul(points_a,np.transpose(points_a))))
    assert R_0.shape[0] == R_0.shape[1] and R_0.shape[1] == 3
    R, _ = np.linalg.qr(R_0) # orthogonalize the R_0
    iter_num = 0

    while abs(np.mean(error(points_a, points_b, R))) > epsilon and iter_num < 400:
        
        iter_num += 1
        points_b_ = np.matmul(np.transpose(R), points_b)
        skew = np.transpose(np.matmul(np.linalg.inv(np.matmul(points_a, np.transpose(points_a))+100.0*np.eye(3)),\
                             np.matmul(points_a, np.transpose(points_b_-points_a))))
        dR = np.eye(3) + skew
        dR, _ = np.linalg.qr(dR)
        R = np.matmul(R, dR)
    return R
    

class Point2Point_Reg(object):
    def __init__(self, *param):
        self.param = param
    
    def _shift(self, points):
        assert isinstance(points, np.ndarray) and len(points.shape) == 2
        avg = np.mean(points, axis=1,keepdims=True)
        return points - avg, avg

    def _compute_rotation(self, points_a, points_b):
        points_a, self.avg_a= self._shift(points_a)
        points_b, self.avg_b= self._shift(points_b)
        R = Direct_solveRotation(points_a, points_b)
        # R = Iterative_solveRotation(points_a, points_b)
        return R

    def _compute_translation(self, points_a, points_b, R):
        return self.avg_b - np.matmul(R, self.avg_a)

    def __call__(self, points_a, points_b):
        """using direct/iteration method to calculate the rotation matrix
            points_a: 2D matrix (3 x N) composed of a sequence of points from A set
            point_b: 2D matrix  (3 x N) composed of a sequence of points from B set
        """
        assert isinstance(points_a, np.ndarray) and points_a.ndim == 2 \
            and isinstance(points_b, np.ndarray) and points_b.ndim == 2
        
        R = self._compute_rotation(points_a, points_b)
        
        t = self._compute_translation(points_a, points_b, R)

        return Cartesian_transformation({'R':R, 't':t})