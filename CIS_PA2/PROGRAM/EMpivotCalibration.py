import numpy as np
from util.Cartesian_transformation import Cartesian_transformation
from util.PointCloudRegistration import Point2Point_Reg
import pandas as pd
from util.pivot_calib import Pivot_calib
from util.CalibrateDistortion import DistortionCalib
from util.LoadSensorData import Compute_expectedC
import os 

def Compute_F(points, Points):
    """
    use registration to compute the cartesian coordinate transformation

    -Input:
        points and Points are 3xN array
    -Return:
        Cartesian_transformation class F which satisfies Points = F(points)
        F.param['R]: Rotation matrix
        F.param['t']: Translation vector
    """
    Reg = Point2Point_Reg()
    return Reg(points, Points)


def EMpivotCalibration(root = './data', status='debug', case='c', dewarping=True, eval = True,calib = None):

    file = os.path.join(root, 'pa2-'+status+'-'+case+'-empivot.txt')
    empivot = pd.read_csv(file)
    if status == 'debug':
        output = pd.read_csv(file.replace('-empivot', '-output1'))
# fit the distortion correction function
    if dewarping and calib == None:
        calib1 = DistortionCalib()
        sensor_data, expected_data = Compute_expectedC('./data',status,case,eval)
        calib1.fit(sensor_data, expected_data)
    elif dewarping and calib != None:
        calib1 = calib
    

# reading empivot 
    heads = [int(c) for c in list(empivot.columns)[:-1]]
    heads = {'N_G': heads[0],'N_Frame':heads[1]}
    # print(heads)
    data = [v for v in empivot.values]
    # print(data)
    Frames_em = []
    for i in range(heads['N_Frame']):
        Frame = data[i * heads['N_G']:(i+1)*heads['N_G']]
        G = Frame
        assert len(G) == heads['N_G']

# rectify the distortion 
        if dewarping:
            G = [calib1.predict(g[:,None])[0] for g in G] 

        Frames_em.append(np.concatenate([g[:,None] for g in G], axis=1))
    assert len(Frames_em) == heads['N_Frame']

# reading answers from known cases
    if status == 'debug':
        heads = [int(c) for c in list(output.columns)[:-1]]
        heads = {'N_C': heads[0],'N_Frame':heads[1]}
        # print(heads)
        data = [v for v in output.values]

        EM_pivot_gt, opt_pivot_gt = data[0], data[1]
        data = data[2:]
        Frames_gt = []
        for i in range(heads['N_Frame']):
            Frame = data[i * heads['N_C']:(i+1)*heads['N_C']]
            C = Frame
            assert len(C) == heads['N_C']
            Frames_gt.append(np.concatenate([d[:,None] for d in C], axis=1))
        assert len(Frames_gt) == heads['N_Frame']
    else:
        EM_pivot_gt = False

#  Compute local Frame g_0
    point_G_1 = Frames_em[0]*1.0
    G_0 = np.mean(point_G_1, axis=1, keepdims=True)
    point_G_fixed = point_G_1 - G_0
    F_G = [Compute_F(point_G_fixed, G) for G in Frames_em if np.sum(Compute_F(point_G_fixed, G).param['R'])!=0] # define [F_11,F_12,F_13,F14, ...]
    
#  Pivot Calibration 
    calib = Pivot_calib()
    p_tip,p_dim = calib(F_G) 
#  Evaluation
    if eval:
        if isinstance(EM_pivot_gt, np.ndarray):
            print('p_dimple: ', p_dim, '    Ground_truth: ', EM_pivot_gt)
        else:
            print('p_dimple: ', p_dim)
        if status == 'debug':
            errors_frames = []
            l2 = lambda a,b: np.linalg.norm(a-b)
            mae = lambda a,b: np.sum(abs(a-b))# compute the mae loss between two column vectors
            print('EM pivot calibration error of case '+case+' : ', l2( p_dim, EM_pivot_gt))
    return p_tip, p_dim, point_G_fixed



if __name__ == '__main__':
    root = './data'
    status='debug'
    case='a'
    dewarping = True
    eval = True
    p_tip, p_dim, point_G_fixed = EMpivotCalibration(root, status, case, dewarping, eval)
    print('p_tip:  ', p_tip)
    # print('g[0] : ', point_G_fixed)
