import numpy as np
from util.Cartesian_transformation import Cartesian_transformation
from scipy import linalg as LA
"""
    In this file, we develop a module for point-cloud-to-point-cloud registration. We implement
    the Point2Point_Reg class with two point sets as input and return a Cartesin_transformation F.
"""

def quaternion_rotation_matrix(Q):
    """
    https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

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
            # print('Error')
            return np.zeros_like(R)
    
def Direct_QuartSolveRotaion(points_a, points_b, epsilon=1e-6):
    # 1. compute H
    H = np.matmul(points_a, np.transpose(points_b))
    assert H.shape[0] == H.shape[1] and H.shape[0] == 3
    # 2. compute G
    delta_T = np.asarray([H[1,2]-H[2,1], H[2,0]-H[0,2], H[0,1]-H[1,0]])
    G_1 = np.concatenate((np.trace(H)[None], delta_T))[None,:]
    G_2 = np.concatenate((delta_T[:, None], H + np.transpose(H) - np.trace(H)*np.eye(3)), axis = 1)
    G = np.concatenate([G_1, G_2], axis = 0)
    # 3. eigen decomposite
    # e_vals, e_vecs = np.linalg.eig(G)
    evals , evecs = LA.eigh(G)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    # 4. eigen vector corresponding to the max eigen value is a unit quaternion
    quaternion = evecs[:,0]
    R = quaternion_rotation_matrix(quaternion)
    return R

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
        # R = R if np.sum(R) > 0.0 else Direct_QuartSolveRotaion(points_a, points_b)
        R = Direct_QuartSolveRotaion(points_a, points_b)
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

if __name__ == '__main__':
    A = np.random.rand(3, 6)
    R = [[1, 0, 0, 1]
         [0, 1, 0, 1]
         [0, 0, 1, 1]]
    T = [[1]
         [1]
         [1]]
    F = [[1, 0, 0, 1]
         [0, 1, 0, 1]
         [0, 0, 1, 1]
         [0, 0, 0, 1]]
    B = np.matmul(R, A) + T

    reg = Point2Point_Reg(A, B)
    # A_q = reg.
