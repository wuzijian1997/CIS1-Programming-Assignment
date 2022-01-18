import numpy as np

class Cartesian_transformation(object):
    def __init__(self, param):
        self.param = param
        

    def inverse(self):
        R = np.transpose(self.param['R'])
        t = - np.matmul(R, self.param['t'])
        return Cartesian_transformation({'R':R,'t':t})

    def __mul__(self, other):
        R = np.matmul(self.param['R'], other.param['R'])
        t = self.param['t'] + np.matmul(self.param['R'], other.param['t'])
        return Cartesian_transformation({'R':R,'t':t})
        

    def __call__(self, p): # p is a 3*1 vector
        assert len(p.shape) == 2 or p.shape[1] != 1
        return np.matmul(self.param['R'], p) + self.param['t']

if __name__ == '__main__':
    param = {}
    param['R']=np.eye(3)
    param['t'] = np.ones((3,1),np.float32)

    F = Cartesian_transformation(param)
    print(F(np.ones((3,1))))
    F1 = F.inverse()
    a = F(np.ones((3,1)))
    print(F1(a))

    F2 = F * F1
    print(F2(np.ones((3,1))))