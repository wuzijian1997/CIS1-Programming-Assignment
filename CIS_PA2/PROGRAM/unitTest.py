import numpy as np
from util.Cartesian_transformation import Cartesian_transformation
from util.CalibrateDistortion import DistortionCalib
"""
unit test of distortion calibration
"""

ground_truth_data = 50 + (200 - 50)*np.random.rand(3,1000)

test_data = 100 + (180 - 100)*np.random.rand(3,200)

#add nonlinear error to fitting data
distorted_sensor_err = [0.001*np.exp(0.002*ground_truth_data[0:1,:])*np.exp(0.005*ground_truth_data[1:2,:])*np.exp(0.01*ground_truth_data[2:3,:]),
                            0.008*np.exp(0.002*ground_truth_data[0:1,:])*np.exp(0.05*ground_truth_data[1:2,:]),
                            0.005*np.exp(0.002*ground_truth_data[0:1,:])*np.exp(0.005*ground_truth_data[2:3,:])]
distorted_sensor_err = 0.2*np.concatenate(distorted_sensor_err, axis=0)
distorted_sensor_data = ground_truth_data + distorted_sensor_err

#add nonlinear error to test_data
distorted_test_err = [0.001*np.exp(0.002*test_data[0:1,:])*np.exp(0.005*test_data[1:2,:])*np.exp(0.01*test_data[2:3,:]),
                           0.008* np.exp(0.002*test_data[0:1,:])*np.exp(0.05*test_data[1:2,:]),
                           0.005* np.exp(0.002*test_data[0:1,:])*np.exp(0.005*test_data[2:3,:])]
distorted_test_data = 0.2*np.concatenate(distorted_test_err, axis=0) + test_data

dewarping_calib = DistortionCalib(order = 5)

dewarping_calib.fit(distorted_sensor_data, ground_truth_data)

corrected_test_data = [dewarping_calib.predict(distorted_test_data[:,i:i+1]) for i in range(distorted_test_data.shape[1])]

corrected_test_data = np.concatenate(corrected_test_data, axis = 0)

l2 = lambda a,b: np.linalg.norm(a-b)

print('error before correction: ', np.mean([l2(test_data[:,i],distorted_test_data[:,i]) for i in range(test_data.shape[1])]))
print('error after correction: ', np.mean([l2(test_data[:,i],corrected_test_data[i,:]) for i in range(test_data.shape[1])]))




