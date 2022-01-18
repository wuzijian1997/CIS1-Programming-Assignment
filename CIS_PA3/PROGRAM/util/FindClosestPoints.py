import numpy as np
import pandas as pd
import os
from optparse import OptionParser
from util.io import read_output, read_surface, result_evaluation, write_output, read_markers_to_body, read_markers_to_tracker
from util.PointCloudRegistration import Point2Point_Reg
from sklearn.neighbors import KDTree
import time

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

class ClosestTriangleSearch(object):

    def __init__(self, *param):
        self.param = param

    def ComputeClosestPoint(self, point, vertex):
        p = np.asarray(vertex[0])
        q = np.asarray(vertex[1])
        r = np.asarray(vertex[2])
        A = np.concatenate([(q-p)[:, None], (r-p)[:, None]], axis = 1)
        B = (np.asarray(point) - np.asarray(vertex[0]))[:, None]
        Lambda, Mu = np.linalg.lstsq(A, B, rcond=None)[0]

        h = p + Lambda * (q - p) + Mu * (r - p)
        if Lambda < 0:
            h = self.ProjectOnSegment(h,r,p)
        elif Mu < 0:
            h = self.ProjectOnSegment(h,p,q)
        elif Lambda + Mu > 1:
            h = self.ProjectOnSegment(h,q,r)
        return h
    
    def ProjectOnSegment(self, c,p,q, return_array = False):
        """
        3x1 ndarray
        """
        if len(c.shape) == 1:
            c,p,q = c[:,None], p[:,None], q[:,None]
        l = np.matmul((c-p).T,  q - p).item()/np.matmul((q-p).T,  q - p).item()
        l = max(0, min(l,1))
        c = p + l*(q-p) if return_array else (p + l*(q-p))[:,0]
        return c

    def ComputeBoudingSphere(self, vertex):
        a = np.asarray(vertex[0])
        b = np.asarray(vertex[1])
        c = np.asarray(vertex[2])
        ab_dist = np.linalg.norm(a - b)
        ac_dist = np.linalg.norm(a - c)
        bc_dist = np.linalg.norm(b - c) 
        dist = np.array([ab_dist, ac_dist, bc_dist])
        max_index = np.argmax(dist)
        if max_index == 0:
            f = (a + b)/2
            u = a - f
            v = c - f
        elif max_index == 1:
            f = (a + c)/2
            u = a - f
            v = b - f
        elif max_index == 2:
            f = (b + c)/2
            u = b - f
            v = a - f
        d = np.cross(np.cross(u, v), u)
        gamma = (np.matmul(v,v) - np.matmul(u,u)) / (2 * np.matmul(d, (v - u)))
        if gamma <= 0:
            l = 0
        else:
            l = gamma
        
        center = f + l * d
        radius = np.linalg.norm(center - a)

        return center, radius

    def BruteForceSearch(self, point_cloud, face, points):
        face = face[0:3]
        c_list = []
        mag_list = []
        
        for i in range(point_cloud.shape[1]):
            a = point_cloud[:,i]
            min_dist = 1.0e5
            for j in range(face.shape[1]):
                vertex_index = face[:, j]
                p = points[:, vertex_index[0]]
                q = points[:, vertex_index[1]]
                r = points[:, vertex_index[2]]
                vertex = np.vstack((p, q, r))
                c = self.ComputeClosestPoint(a, vertex)
                if np.linalg.norm(a-c) < min_dist:
                    min_dist = np.linalg.norm(a-c)
                    h = c
            c_list.append(h)
            mag = np.linalg.norm(a-h)
            mag_list.append(mag)
        c_calc = np.transpose(np.asarray(c_list))
        mag_calc = np.transpose(np.asarray(mag_list))
        return c_calc, mag_calc

    def BoundingSphereSearch(self, point_cloud, face, points):
        face = face[0:3]
        c_list = []
        mag_list = []
        rhos_list = []
        centers_list = []
        # compute bounding sphere
        for j in range(face.shape[1]):
            vertex_index = face[:, j]
            p = points[:, vertex_index[0]]
            q = points[:, vertex_index[1]]
            r = points[:, vertex_index[2]]
            vertex = np.vstack((p, q, r))
            center, rho = self.ComputeBoudingSphere(vertex)
            centers_list.append(center)
            rhos_list.append(rho)

        # calculate
        for i in range(point_cloud.shape[1]):
            bound = 1.0e5 # initial bound
            a = point_cloud[:,i]
            for j in range(face.shape[1]):
                # get vertex of a triangle
                vertex_index = face[:, j]
                p = points[:, vertex_index[0]]
                q = points[:, vertex_index[1]]
                r = points[:, vertex_index[2]]
                vertex = np.vstack((p, q, r))

                if np.linalg.norm(centers_list[j] - a) - rhos_list[j] <= bound:
                    h = self.ComputeClosestPoint(a, vertex)
                    if np.linalg.norm(h - a) < bound:
                        bound = np.linalg.norm(h - a)
                        c = h

            c_list.append(c)
            mag = np.linalg.norm(a-c)
            mag_list.append(mag)

            #c_list = np.vstack((c_list,c))
        c_calc = np.transpose(np.asarray(c_list))
        mag_calc = np.transpose(np.asarray(mag_list))
        return c_calc, mag_calc

    def KDTreeSearch(self, point_cloud, face, points):
        face = face[0:3]
        c_list = []
        mag_list = []
        h_list = []
        
        for i in range(point_cloud.shape[1]):
            a = point_cloud[:,i]
           
            for j in range(face.shape[1]):
                vertex_index = face[:, j]
                p = points[:, vertex_index[0]]
                q = points[:, vertex_index[1]]
                r = points[:, vertex_index[2]]
                vertex = np.vstack((p, q, r))
                c = self.ComputeClosestPoint(a, vertex)
                c_list.append(c)
            
            c_array = np.asarray(c_list)
            
            tree = KDTree(c_array, leaf_size=2)
            
            a = np.transpose(a[:,None])
            
            dist, ind = tree.query(a, k=1)
            

            h = np.squeeze(c_array[ind])
            h_list.append(h)
            mag = np.linalg.norm(a-h)
            mag_list.append(mag)
        c_calc = np.transpose(np.asarray(h_list))
        mag_calc = np.transpose(np.asarray(mag_list))
        return c_calc, mag_calc

    def __call__(self, point, vertex):
        h = self.ComputeClosestPoint(point, vertex)
        return h

if __name__=='__main__':

    ## test ComputeClosestPoint(self, point, vertex)
    point = [1, 1, 1]
    verter = [[-1,  0,  0],
              [ 1,  1,  0],
              [ 1, -1,  0]]
    closest_operator = ClosestTriangleSearch()
    h = closest_operator(point, verter)
    #print(h)

    points, face = read_surface('./data/Problem3Mesh.sur')

    ## test ComputeBoudingSphere(self, vertex)
    verter_tri = [[ 0,  0,  0],
                  [ 1,  1,  0],
                  [ 2,  0,  0]]
    center, radius = ClosestTriangleSearch().ComputeBoudingSphere(verter_tri)
    print(center, radius)

    ## test run.py - get d_k
    parser = OptionParser()

    parser.add_option("--root", dest="root", help="the folder containing all the data", default='./data', type="string")
    parser.add_option("--status", dest="status", help="Debug or Unknown", default='Debug', type="string")
    parser.add_option("--case", dest="case", help=" the case ID of data file, e.g. A, B, C, D ...", default='B', type="string")
    parser.add_option("-e", help= " evaluate the results of subroutines ", action="store_true", dest="eval")
    parser.add_option("-q", help= " close the evaluation of subroutine results ", action="store_false", dest="eval", default=False)

    (options, args) = parser.parse_args()

    root = options.root
    case = options.case
    status = options.status

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
    print(points_d.shape)
    #print(np.transpose(points_d))

    ## test BoundingSphereSearch(self, point_cloud, face, points)
    c_calc, mag_calc = ClosestTriangleSearch().BoundingSphereSearch(points_d, face, points)
    print(c_calc.shape)
    print(mag_calc)

    ## evaluation
    d_gt, c_gt, mag = read_output('./data/PA3-B-Debug-Output.txt')
    
    print(mag)

    d_L2 = result_evaluation(d_gt, points_d)
    c_L2 = result_evaluation(c_gt, c_calc)
    mag_L2 = result_evaluation(mag, mag_calc)
    print(d_L2)
    print(c_L2)
    print(mag_L2)

    ## test BruteForceSreach()
    c_calc_b, mag_calc_b = ClosestTriangleSearch().BoundingSphereSearch(points_d, face, points)
    print(c_calc_b.shape)
    print(mag_calc_b)
    
    d_L2 = result_evaluation(d_gt, points_d)
    c_L2 = result_evaluation(c_gt, c_calc_b)
    mag_L2 = result_evaluation(mag, mag_calc_b)
    print(d_L2)
    print(c_L2)
    print(mag_L2)

