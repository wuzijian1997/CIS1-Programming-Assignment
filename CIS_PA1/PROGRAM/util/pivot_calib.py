import numpy as np
from util.Cartesian_transformation import Cartesian_transformation
import math
# from PointCloudRegistration import Point2Point_Reg

class Pivot_calib(object):
    def __init__(self, *param):
        self.param = param
    
    def __call__(self, transforms):
        # for i in range(8):
        #         print(np.concatenate([transforms[i].param['R'], -1*np.eye(3)], axis=1))
        A = np.concatenate(
                [np.concatenate(\
                    [transform.param['R'], -1*np.eye(3)], axis=1) for transform in transforms], axis=0)
        assert A.shape[0] == 3*len(transforms) and A.shape[1] == 6 and A.ndim == 2
        B = np.concatenate([-1* transform.param['t'] for transform in transforms], axis=0)
       
        x = np.linalg.lstsq(A, B)[0]
        return x[3:,0]

if __name__=='__main__':
        calib = Pivot_calib()
        a_list = np.linspace(0,math.pi/2.0,10)
        R = lambda a :np.array([[math.cos(a), math.sin(a),0],
                [-1*math.sin(a), math.cos(a),0],
                [0, 0,1]])
        # print(np.linalg.det(R))
        # p_tip = np.array(4,3,8)
        Fs = [Cartesian_transformation({'R':np.eye(3),'t':np.array([[1],[1],[0]])}),\
                Cartesian_transformation({'R':R(math.pi/6),'t':np.array([[1.5],[math.sqrt(3)/2.0],[0]])})]
        print(calib(Fs))