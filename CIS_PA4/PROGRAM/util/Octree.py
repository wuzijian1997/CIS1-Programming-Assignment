import numpy as np
import pandas as pd
import os
from optparse import OptionParser
from util.PointCloudRegistration import Point2Point_Reg
from util.io import *
import sys
sys.setrecursionlimit(14**7) # max depth of recursion

class BoundingSphere(object):
    def __init__(self, vertex,i):
        """
        Parameters
        -----------
        mesh_points: 3x3 array with cornor points as row vectors
        """
        self.index = i
        self.vertex = vertex
        self.center, self.radius = self.ComputeBoudingSphere(vertex)

    def ProjectOnSegment(self,c,p,q, return_array = False):
        """
        3x1 ndarray
        Return
        ---------
        c: 1d array
        """
        if len(c.shape) == 1:
            c,p,q = c[:,None], p[:,None], q[:,None]
        l = np.matmul((c-p).T,  q - p).item()/np.matmul((q-p).T,  q - p).item()
        l = max(0, min(l,1))
        c = p + l*(q-p) if return_array else (p + l*(q-p))[:,0]
        return c

    def ComputeBoudingSphere(self, vertex):
        """
        Parameters
        ----------
        vertex:3x3 array with cornor points as row vectors
        """
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

    def ComputeClosestPoint(self, point):
        vertex = self.vertex
        p = np.asarray(vertex[0])
        q = np.asarray(vertex[1])
        r = np.asarray(vertex[2])
        A = np.concatenate([(q-p)[:, None], (r-p)[:, None]], axis = 1)
        B = (np.asarray(point) - np.asarray(vertex[0]))[:, None]
        Lambda, Mu = np.linalg.lstsq(A, B,rcond=None)[0]

        h = p + Lambda * (q - p) + Mu * (r - p)
        if Lambda < 0:
            h = self.ProjectOnSegment(h,r,p)
        elif Mu < 0:
            h = self.ProjectOnSegment(h,p,q)
        elif Lambda + Mu > 1:
            h = self.ProjectOnSegment(h,q,r)
        return h


class BoundingBoxTreeNode(object):
    def __init__(self, Spheres, nSpheres):
        self.Spheres = Spheres
        self.nSpheres = nSpheres
        self.SplitPoint = self.Centroid(Spheres)
        self.MaxRadius = self.FindMaxRadius(Spheres)
        self.UB = self.FindMaxCoordinates(Spheres)
        self.LB = self.FindMinCoordinates(Spheres)
        # self.Subtrees = [[[[] for i in range(2)] for j in range(2)] for k in range(2)]
        self.Subtrees = np.ndarray((2,2,2), dtype=BoundingBoxTreeNode)
        self.ConstructSubtrees()

    def ConstructSubtrees(self, minCount = 20):
        if self.nSpheres <= minCount:
            self.HaveSubtrees = False
            return
        self.HaveSubtrees = True
        SubSpheres = self.Split(self.SplitPoint, self.Spheres)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    spheres = SubSpheres[i][j][k]
                    self.Subtrees[i][j][k] = BoundingBoxTreeNode(spheres, len(spheres))
                    # assert len(self.Subtrees[i][j][k]) == 0
                    # self.Subtrees[i][j][k].append(BoundingBoxTreeNode(spheres, len(spheres)))

    def FindClosestPoint(self, v, param):#bound, closest):
        """
        Recursively find the closest-to-vector point in bounding spheres of the tree

        Parameters
        -----------
        v: 1D array.
        param['bound']: double.
        param['closest']: 1d array. coordinates of the closest point on triangles
        """
        # self.param = param
        if len(self.Spheres) == 0:
            return
        dist = self.MaxRadius + param['bound']
        if (v - self.UB > dist).any() or (self.LB - v > dist).any(): return
        if self.HaveSubtrees:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # self.Subtrees[i][j][k][0].FindClosestPoint(v,param)
                        self.Subtrees[i,j,k].FindClosestPoint(v,param)
        else:
            for S in self.Spheres:
                self.UpdateClosest(S,v, param)

    def UpdateClosest(self, sphere, v, param):
        dist = np.linalg.norm(v - sphere.center)
        if dist - sphere.radius > param['bound']: return
        # param['num'] += 1
        c = sphere.ComputeClosestPoint(v)
        dist = np.linalg.norm(c - v)
        if dist < param['bound']:
            param['bound'] = dist
            param['closest'] = c

    def Split(self, SplitPoint, Spheres):
        """
        Parameters
        ----------
        SplitPoint: 1d array
        Spheres : list of BoundingSpheres

        Return
        -------
        SubSpheres: 3D list of spheres splited into 8 quardrants. e.g. SubSpheres[False][True][True]
        """
        SubSpheres = [[[[] for i in range(2)] for j in range(2)] for k in range(2)]


        if len(Spheres) == 2:
            if np.linalg.norm(Spheres[0].center - Spheres[0].center) < 1e-6:
                SubSpheres[0][0][0].append(Spheres[0])
                SubSpheres[1][1][1].append(Spheres[1])
                return SubSpheres
        for S in Spheres:
           i = SplitPoint[0] < S.center[0]
           j = SplitPoint[1] < S.center[1]
           k = SplitPoint[2] < S.center[2]
           SubSpheres[i][j][k].append(S)

        return SubSpheres

    def Centroid(self,Spheres):
        if len(Spheres) == 0:
            return None
        centers = np.vstack([S.center for S in Spheres])
        return np.mean(centers, axis = 0)

    def FindMaxCoordinates(self, Spheres):
        if len(Spheres) == 0:
            return None
        UBs = np.vstack([S.center + S.radius for S in Spheres])
        UB = np.max(UBs, axis=0)
        return UB

    def FindMinCoordinates(self, Spheres):
        if len(Spheres) == 0:
            return None
        LBs = np.vstack([S.center - S.radius for S in Spheres])
        LB = np.min(LBs, axis=0)
        return LB

    def FindMaxRadius(self, Spheres):
        if len(Spheres) == 0:
            return None
        return max([S.radius for S in Spheres])


if __name__ == '__main__':
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
    parser = OptionParser()

    parser.add_option("--root", dest="root", help="the folder containing all the data", default='./data', type="string")
    # parser.add_option("--status", dest="status", help="Debug or Unknown", default='Debug', type="string")
    parser.add_option("--case", dest="case", help=" the case ID of data file, e.g. A, B, C, D ...", default='A', type="string")
    parser.add_option("-e", help= " evaluate the results of subroutines ", action="store_true", dest="eval")
    parser.add_option("-q", help= " close the evaluation of subroutine results ", action="store_false", dest="eval", default=False)

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

    points, faces = read_surface('./data/Problem3Mesh.sur')

    points, faces = points.T, faces.T
    boundingSphere_list = []
    for i in range(len(faces)):
        id1 = faces[i][0]
        id2 = faces[i][1]
        id3 = faces[i][2]
        vertexes = np.vstack([points[id1],points[id2],points[id3]])
        boundingSphere_list.append(BoundingSphere(vertexes,i))

    Tree = BoundingBoxTreeNode(boundingSphere_list, len(boundingSphere_list))
    point_d = points_d[:,0]
    param = {}
    param['bound'] = 10000
    Tree.FindClosestPoint(point_d,param)
    c = param['closest']
    print(c)
    print(param['bound'])



