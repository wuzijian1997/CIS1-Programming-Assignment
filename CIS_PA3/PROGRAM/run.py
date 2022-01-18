import pandas as pd
import os
from optparse import OptionParser
import numpy as np
from util.PointCloudRegistration import Point2Point_Reg
from util.io import read_output, read_surface, result_evaluation, write_output, read_markers_to_body, read_markers_to_tracker
from util.FindClosestPoints import ClosestTriangleSearch
import time
from util.Octree import BoundingBoxTreeNode, BoundingSphere

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

def OctreeSearch(points, face):
    """
    calculate the closest point using Octree
    """
    points, faces = points.T, face.T
    boundingSphere_list = []
    for i in range(len(faces)):
        id1 = faces[i][0]
        id2 = faces[i][1]
        id3 = faces[i][2]
        vertexes = np.vstack([points[id1],points[id2],points[id3]])
        boundingSphere_list.append(BoundingSphere(vertexes,i))

    # Construct the octree
    Tree = BoundingBoxTreeNode(boundingSphere_list, len(boundingSphere_list))
    c = []
    mag = []
    for i in range(points_d.shape[1]):
        #calculate the closest point recusively
        Tree.FindClosestPoint(points_d[:,i],{'bound':1000})
        cp = Tree.param['closest']
        c.append(cp)
        mag.append(np.linalg.norm(cp - points_d[:,i]))
    c = np.asarray(c)
    mag = np.asarray(mag)
    return c,mag

if __name__=='__main__':

    parser = OptionParser()

    parser.add_option("--root", dest="root", help="the folder containing all the data", default='./data', type="string")
    parser.add_option("--case", dest="case", help=" the case ID of data file, e.g. A, B, C, D ...", default='A', type="string")
   

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
    Abody_file, Bbody_file = os.path.join(root, 'Problem3-BodyA.txt'), os.path.join(root, 'Problem3-BodyB.txt')
    A_coords, A_tip = read_markers_to_body(Abody_file)
    B_coords, B_tip = read_markers_to_body(Bbody_file)

    # read a, b points with respect to optical tracker
    file_name = os.path.join(root, 'PA3-' + case + '-' + status + '-SampleReadingsTest.txt')
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
    points, face = read_surface('./data/Problem3Mesh.sur')

    ## Brute-Force Search
    start = time.time()
    c_calc, mag_calc = ClosestTriangleSearch().BruteForceSearch(points_d, face, points)
    end = time.time()
    print("Case: "+case+'.'," Brute-Force Search time = %.3f s" %(end - start))

    ## Bounding Sphere Search
    start = time.time()
    c_calc_b, mag_calc_b = ClosestTriangleSearch().BoundingSphereSearch(points_d, face, points)
    end = time.time()
    print("Case: "+case+'.'," Bounding Sphere Search time = %.3f s" %(end - start))

    ## Octree Search
    start = time.time()
    c_calc_octree, mag_calc_octree = OctreeSearch(points, face)
    end = time.time()
    print("Case: "+case+'.'," OcTree Search time = %.3f s" %(end - start))

    ## KDTree Search
    start = time.time()
    c_calc_k, mag_calc_k = ClosestTriangleSearch().KDTreeSearch(points_d, face, points)
    end = time.time()
    print("Case: "+case+'.'," KDTree Search time = %.3f s" %(end - start))
    


    ## Results evaluation
    if status == 'Debug':
        d_gt, c_gt, mag = read_output(os.path.join(root,'PA3-'+case+'-Debug-Output.txt'))
        d_L2 = result_evaluation(d_gt, points_d)

        c_L2 = result_evaluation(c_gt, c_calc)
        mag_L2 = result_evaluation(mag, mag_calc)
        print("BrutalSearch:" + " d mean L2 = ", d_L2)
        print("BrutalSearch:" + " c mean L2 = ",c_L2)
        print("BrutalSearch:" + " mag mean L2 = ", mag_L2)

        c_L2 = result_evaluation(c_gt, c_calc_k)
        mag_L2 = result_evaluation(mag, mag_calc_k)
        print("KdTree:" + " d mean L2 = ", d_L2)
        print("KdTree:" + " c mean L2 = ",c_L2)
        print("KdTree:" + " mag mean L2 = ", mag_L2)

        c_L2 = result_evaluation(c_gt, c_calc_b)
        mag_L2 = result_evaluation(mag, mag_calc_b)
        print("BoundingSphereSearch:" + " d mean L2 = ", d_L2)
        print("BoundingSphereSearch:" + " c mean L2 = ",c_L2)
        print("BoundingSphereSearch:" + " mag mean L2 = ", mag_L2)


        c_L2 = result_evaluation(c_gt, c_calc_octree.T)
        mag_L2 = result_evaluation(mag, mag_calc_octree)
        print("OcTree:" + " d mean L2 = ", d_L2)
        print("OcTree:" + " c mean L2 = ",c_L2)
        print("OcTree:" + " mag mean L2 = ", mag_L2)



    ## Generate output file
    root = '../OUTPUT'
    filename = os.path.join(root, 'PA3-' + case + '-' + status + '-Output.txt')
    data = np.vstack((points_d, c_calc, mag_calc))
    write_output(filename, data)