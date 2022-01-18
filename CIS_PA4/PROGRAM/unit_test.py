import pandas as pd
import os
from optparse import OptionParser
import numpy as np
from util.PointCloudRegistration import Point2Point_Reg
from util.io import read_output, read_surface, result_evaluation, write_output, read_markers_to_body, read_markers_to_tracker
import time
from ICP import ICP

np.random.seed(123)
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

def add_noise(points, noisy_ratio, noisy_level):
    for i in range(points.shape[1]):
        if np.random.rand() < noisy_ratio:
            points[:,i] += noisy_level * (-1 + 2*np.random.rand(3))
    return points

if __name__=='__main__':

    parser = OptionParser()

    parser.add_option("--root", dest="root", help="the folder containing all the data", default='./data', type="string")
    parser.add_option("--case", dest="case", help=" the case ID of data file, e.g. A, B, C, D ...", default='B', type="string")
    parser.add_option("-r", help= " eliminate the incorrect matching",action="store_false", dest="robust", default=False)

    (options, args) = parser.parse_args()

    root = options.root
    case = options.case
    # status = options.status
    files = os.listdir(root)
    for file in files:
        file = file.split('/')[-1]
        if '-'+case+'-' in file:
            status = file.split('-')[-2] # Find the status (Unknown or Debug) for the dataset

    # read A, B points with respect to rigid body frame
    Abody_file, Bbody_file = os.path.join(root, 'Problem4-BodyA.txt'), os.path.join(root, 'Problem4-BodyB.txt')
    A_coords, A_tip = read_markers_to_body(Abody_file)
    B_coords, B_tip = read_markers_to_body(Bbody_file)

    # read a, b points with respect to optical tracker
    file_name = os.path.join(root, 'PA4-' + case + '-' + status + '-SampleReadingsTest.txt')
    a_frames, b_frames = read_markers_to_tracker(file_name)
    N_Frame = len(a_frames)

    points_d = [] 
    for k in range(N_Frame):
        points_a, points_b = a_frames[k], b_frames[k]
        F_A = Compute_F(A_coords, points_a)
        F_B = Compute_F(B_coords, points_b)
        F = F_B.inverse() * F_A
        points_d.append(F(A_tip))

    points_d = np.concatenate(points_d, axis = 1)


    ## Get surface information
    points, face = read_surface('./data/Problem4MeshFile.sur')

    start = time.time()
    icp = ICP(threshold = [0.01]*3, searcher = 'Octree', robust=True)
    noisy_points = add_noise(points, noisy_ratio=1.0,noisy_level=3)
    F,closest_pts = icp(points_d,noisy_points,face)
    end = time.time()
    points_s = F(points_d)

    # print(points_s.T)
    mag_diff = np.linalg.norm(points_s.T - closest_pts.T, axis=1)
    print(' time:', end-start)

    
    ## Results evaluation
    if status == 'Debug':
        s_gt, c_gt, mag = read_output(os.path.join(root,'PA4-'+case+'-Debug-Answer.txt'))
        s_L2 = result_evaluation(s_gt, points_s)
        c_L2 = result_evaluation(c_gt, closest_pts)
        mag_L2 = result_evaluation(mag, mag_diff)
        print(" s error = ", s_L2)
        print(" c error = ", c_L2)
        print(" mag error = ", mag_L2)
