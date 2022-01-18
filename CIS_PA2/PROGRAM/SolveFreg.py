import numpy as np
from util.PointCloudRegistration import Point2Point_Reg
import pandas as pd
import os
from util.LoadSensorData import Compute_expectedC
from util.CalibrateDistortion import DistortionCalib
from EMpivotCalibration import Compute_F, EMpivotCalibration

def SolveFreg(root='./data', status = 'debug', case = 'e', dewarping = True, eval = False, calib=None):

    em_fid = os.path.join(root, 'pa2-' + status + '-' + case + '-em-fiducialss.txt')
    ct_fid = os.path.join(root, 'pa2-' + status + '-' + case + '-ct-fiducials.txt')
    em_fid = pd.read_csv(em_fid)
    ct_fid = pd.read_csv(ct_fid)

# fit the distortion correction function
    if dewarping and calib == None:
        calib1 = DistortionCalib()
        sensor_data, expected_data = Compute_expectedC('./data',status,case,eval)
        calib1.fit(sensor_data, expected_data)
    elif dewarping and calib != None:
        calib1 = calib
    else:
        calib1 = None


# read data from em-fiducialss
    if '6.1' in list(em_fid.columns): # a bug in reading char '6' as '6.1'
        heads = [int(c) for c in list(em_fid.columns)[:-2]]
        heads.append(6)
    else:
        heads = [int(c) for c in list(em_fid.columns)[:-1]]
    heads = {'N_G': heads[0],'N_Frame':heads[1]}
    # print(heads)
    data = [v for v in em_fid.values]
    # print(data)
    Frames_em = []
    for i in range(heads['N_Frame']):
        G = data[i * heads['N_G']:(i+1) * heads['N_G']]
        assert len(G) == heads['N_G']
    # rectify the distortion 
        if dewarping:
            G = [calib1.predict(g[:,None])[0] for g in G] 
        Frames_em.append(np.concatenate([g[:,None] for g in G], axis=1))
    assert len(Frames_em) == heads['N_Frame']

# read data from ct-fiducialss
    data = [v[:,None] for v in ct_fid.values]
    b = np.concatenate(data, axis=1)
    assert len(Frames_em) == 6

# EM pivot calibration
    p_tip, p_dim, point_G_fixed = EMpivotCalibration(root, status,case,dewarping,eval,calib=calib1)
    F_Gs = [Compute_F(point_G_fixed, G) for G in Frames_em]

# get points set {B}
    B = [F_G(p_tip[:,None]) for F_G in F_Gs]
    B = np.concatenate(B,axis = 1) # 3 x N 
# get Freg by using registration from B to b
    Freg = Compute_F(B, b)
    return Freg


if __name__=='__main__':
    F = SolveFreg(root='./data', status='debug', case='b', dewarping = True, eval=False, calib = None)