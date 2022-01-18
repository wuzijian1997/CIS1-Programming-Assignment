import numpy as np
from util.PointCloudRegistration import Point2Point_Reg
import pandas as pd
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



def Compute_expectedC(root = './data', status='debug', case='a',eval=True):
    """
    Compute the expected C and the recorded C of EM racker
    """
    calbody = pd.read_csv(os.path.join(root,'pa2-'+status+'-'+case+'-calbody.txt'))
    calreadings = pd.read_csv(os.path.join(root,'pa2-'+status+'-'+case+'-calreadings.txt'))
    if status == 'debug':
        output = pd.read_csv(os.path.join(root,'pa2-'+status+'-'+case+'-output1.txt'))

# reading fixed markers on local coordinate system
    heads = [int(c) for c in list(calbody.columns)[:-1]]
    heads = {'N_D': heads[0],'N_A':heads[1],'N_C':heads[2]}
    data = [v[:3] for v in calbody.values]
    points_d, points_a, points_c= data[:heads['N_D']], data[heads['N_D']:heads['N_D']+heads['N_A']], data[heads['N_D']+heads['N_A']:]
    points_d = np.concatenate([point_d[:,None] for point_d in points_d], axis = 1)
    points_a = np.concatenate([point_a[:,None] for point_a in points_a], axis = 1)
    points_c = np.concatenate([point_c[:,None] for point_c in points_c], axis = 1)


# reading frames data
    if ' 8.1' in list(calreadings.columns): # a bug in reading char '8' as '8.1'
        heads = [int(c) for c in list(calreadings.columns)[:-2]]
        heads.append(8)
    else:
        heads = [int(c) for c in list(calreadings.columns)[:-1]]

    heads = {'N_D': heads[0],'N_A':heads[1],'N_C':heads[2], 'N_Frame':heads[3]}
    data = [v[:3] for v in calreadings.values]
    Frames = []
    for i in range(heads['N_Frame']):
        Frame = data[i*(heads['N_D']+heads['N_A']+heads['N_C']):(i+1)*(heads['N_D']+heads['N_A']+heads['N_C'])]
        D, A, C = Frame[:heads['N_D']], Frame[heads['N_D']:heads['N_D']+heads['N_A']], Frame[heads['N_D']+heads['N_A']:]
        assert len(C) == heads['N_C'] and len(A) == heads['N_A'] and len(D) == heads['N_D']
            
        Frames.append([np.concatenate([d[:,None] for d in F], axis=1) for F in [D,A,C]])
    assert len(Frames) == heads['N_Frame']


#  ================================= Q-4-a. Compute the F_D for each Frame ====================================
    """
        points_d:   point set {d_i}
        Frame[0]:   point set {D_i} in one certain Frame
        F_D_frames: A list of Cartesian_transformation class objects containing F_D computed in different frames
    """
    F_D_frames = [Compute_F(points_d, Frame[0]) for Frame in Frames]
    # print('F_D in the first transformation:', F_D_frames[0].param)  


#  ================================= Q-4-b. Compute the F_A for each Frame ====================================
    """
        points_a:   point set {a_i}
        Frame[1]:   point set {A_i} in one certain Frame
        F_A_frames: A list of Cartesian_transformation class objects containing F_A computed in different frames
    """
    F_A_frames = [Compute_F(points_a, Frame[1]) for Frame in Frames]
    # print('F_A in the first transformation:', F_A_frames[0].param)

#  ================================= Q-4-c. Compute the C_expect for each Frame =================================
    """
        points_a:   point set {a_i}
        Frame[1]:   point set {A_i} in one certain Frame
        F_A_frames: A list of Cartesian_transformation class objects containing F_A computed in different frames
    """
    F_DA_frames = [F_D.inverse() * F_A for F_D, F_A in zip(F_D_frames, F_A_frames)]
    # print('F_DA at the first frame:')
    # print(F_DA_frames[0].param)
    C_expects_frames = []
    for F_DA in F_DA_frames:
        C_expects = np.zeros_like(points_c)
        for i in range(C_expects.shape[1]):
            C_expects[:,i] = F_DA(points_c[:,i][:,None])[:,0]
        C_expects_frames.append(C_expects)
    # if eval:
    #     print('Expected C points for all frames:')
    #     pred_dict = {'C coords for Frame {0:}:'.format(k): np.transpose(f) for k,f in zip(range(len(C_expects_frames)),C_expects_frames)}
    #     print(pred_dict)
#  ================================= Q-4-d. Evaluate the accuracy and output the C_expection for each frame =================================
    
# reading ground-truth from debug cases
    if status == 'debug':
        heads = [int(c) for c in list(output.columns)[:-1]]
        heads = {'N_C': heads[0],'N_Frame':heads[1]}
        # print(heads)
        data = [v[:3] for v in output.values]
        data = data[2:]
        Frames_gt = []
        for i in range(heads['N_Frame']):
            Frame = data[i * heads['N_C']:(i+1)*heads['N_C']]
            C = Frame
            assert len(C) == heads['N_C']
            Frames_gt.append(np.concatenate([d[:,None] for d in C], axis=1))
        assert len(Frames_gt) == heads['N_Frame']
        
    # evaluate for debug cases
        errors_frames = []
        l2 = lambda a,b: np.linalg.norm(a-b)
        mae = lambda a,b: np.sum(abs(a-b)) # compute the mae loss between two column vectors
        # print(np.transpose(Frames_gt[0]))
        for C_expects, C_gts in zip(C_expects_frames, Frames_gt):
            assert C_expects.shape == C_gts.shape
            errors = np.asarray([l2(C_expects[:,i:i+1], C_gts[:,i:i+1]) for i in range(C_gts.shape[1])], np.float32)

            errors_frames.append(np.mean(errors))
        
    if status == 'debug' and eval:
        # print('Expected C points:', np.transpose(C_expects_frames[0]))
        # print('Ground truth C points:', np.transpose(Frames_gt[0]))
        print('C_expected error for each frame:         ', errors_frames)
        print('C_expected average error for all frames:  ', np.mean(errors_frames))
    C_sensor_frames = [Frame[2] for Frame in Frames]
    C_sensor = np.concatenate(C_sensor_frames, axis=1)
    C_expects = np.concatenate(C_expects_frames, axis=1)
    assert C_sensor.shape == C_expects.shape
    return C_sensor, C_expects


