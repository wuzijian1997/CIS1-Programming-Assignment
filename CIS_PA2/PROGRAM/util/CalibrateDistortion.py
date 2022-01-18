import numpy as np
from util.PointCloudRegistration import Point2Point_Reg
import pandas as pd
from util.LoadSensorData import Compute_expectedC
from util.Cartesian_transformation import Cartesian_transformation


class DistortionCalib(object):
    def __init__(self, order=5):
        self.order = order
        self.basis_func = lambda i,u : np.asarray(np.math.factorial(i),np.float32)/np.math.factorial(order) * u**i * (1-u)**(order-i)
        self.polynomial_func = lambda i,j,k,u: self.basis_func(i,u[0]) * self.basis_func(j,u[1])* self.basis_func(k,u[2])

    def scale(self, points):
        """
        normalize the points into [0,1]
        """
        self.min_vals = np.min(points, axis = 1, keepdims=True)
        self.max_vals = np.max(points, axis = 1, keepdims=True)
        scale_points = (points - self.min_vals)/(self.max_vals - self.min_vals)
        assert np.max(scale_points) <=1 and np.min(scale_points) >= 0
        return scale_points

    def polynomial(self, u):
        """
        return the [F_000(u),...,F_ijk(u), ..., F_555(u)] row vector
        """
        assert isinstance(u, np.ndarray) and  u.size == 3
        if len(u.shape) > 1:
            u = u.squeeze()
        F_u = []
        for i in range(self.order+1):
            for j in range(self.order+1):
                for k in range(self.order+1):
                    F_u.append(self.polynomial_func(i,j,k,u))
        F_u = np.asarray(F_u, np.float32)[None,:]
        return F_u

    def fit(self, sensor_data, ground_truth):
        """
        use least-square method to fit the correction function based on distorted sensor data and groundtruth

        Parameters
        ----------
        sensor_data: 3xN ndarray 
        ground_truth: 3xN ndarray

        Return
        ----------
        C: Nx3 ndarray, fitted coefficients of polynomials 

        """
        assert isinstance(ground_truth, np.ndarray) and len(ground_truth.shape) == 2
        if ground_truth.shape[0] == 3:
            ground_truth = np.transpose(ground_truth)
        sensor_data = self.scale(sensor_data)
        F = np.concatenate([self.polynomial(sensor_data[:, i]) \
                    for i in range(sensor_data.shape[1])], axis = 0)
        self.C = np.linalg.lstsq(F, ground_truth)[0]
        return self.C
    
    def predict(self, y):
        """
        -Input:
            point: 3 x 1 ndarray denoting the coordination of a sensored point
        """
        assert len(y.shape) == 2
        y = (y - self.min_vals)/(self.max_vals - self.min_vals)
        u = self.polynomial(np.transpose(y))
        return np.matmul(u , self.C)



if __name__=='__main__':

    # obtain sensor data {C}, fixed points {c} with respect to local coordinate system, F_D and F_A for each frame
    sensor_data, expected_data = Compute_expectedC('./data', 'debug', 'b')
    frame_num = len(sensor_data)
    calib = DistortionCalib()
    calib.fit(sensor_data, expected_data)
    num = 0
    calib_data = []
    for i in range(expected_data.shape[1]):
        calib_data.append(calib.predict(sensor_data[:,i][:, None]))

    calib_data = np.concatenate(calib_data, axis = 0)
    print(calib_data)
    

        

    