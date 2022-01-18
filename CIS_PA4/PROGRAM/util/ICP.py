from util.Cartesian_transformation import Cartesian_transformation
import numpy as np
from util.PointCloudRegistration import Point2Point_Reg
from util.io import read_output, read_surface, result_evaluation, write_output, read_markers_to_body, read_markers_to_tracker
from util.FindClosestPoints import SimpleBoundingSphere, BruteForce
from util.Octree import BoundingBoxTreeNode, BoundingSphere

searcher_dict = {'Octree':BoundingBoxTreeNode,
    'BruteForce': BruteForce,
    'BoundingSphere': SimpleBoundingSphere}
class ICP(object):
    def __init__(self, threshold,searcher,robust=True):
        self.threshold = threshold
        self.searcher = searcher
        self.robust = robust
        self.param = {}

    def find_best_transform(self,points, Points,mask):
        """
        use registration to compute the cartesian coordinate transformation

        Parameter
        ----------
            points and Points are 3xN array
            mask: mask for the selected valid matching
        Return
        ----------
            Cartesian_transformation class F which satisfies Points = F(points)
            F.param['R]: Rotation matrix
            F.param['t']: Translation vector
        """
        Reg = Point2Point_Reg()
        F = Reg(points[:,mask], Points[:,mask])
        residual_err = F(points) - Points
        err = np.linalg.norm(residual_err, axis=0)
        self.param['residual_err'] = err
        
        self.param['lst_err_avg'] = self.param['err_avg'] if 'err_avg' in self.param else 0

        valid_err = err[mask]
        self.param['err_avg'] = np.mean(valid_err)
        self.param['err_max'] = np.max(valid_err)
        self.param['sigma'] = np.sqrt(np.sum(valid_err**2))/len(valid_err)
        return F
        

    def read_surf(self, points, face):
        """
        read the surface model file .sur
        """
        points, faces = points.T, face.T
        self.boundingSphere_list = []
        for i in range(len(faces)):
            id1 = faces[i][0]
            id2 = faces[i][1]
            id3 = faces[i][2]
            self.vertexes = np.vstack([points[id1],points[id2],points[id3]])
            self.boundingSphere_list.append(BoundingSphere(self.vertexes,i))
        assert len(self.vertexes) > 0
        return self.vertexes

    def search(self, points, F, searcher):
        """
        call searcher to search the closest needlepoint on surface model for each point in points cloud

        Parameters
        ----------
            points: point cloud
            F: SE(3) transformation
        Return
        -------
            closest_pts: 3xN ndarray, array of closest points 
        """
        closest_pts = []
        points = F(points)
        mask = np.ones(points.shape[1]) > 0
        for i in range(points.shape[1]):
            #calculate the closest point recusively
            pt = points[:,i]
            self.param['bound'] = self.param['residual_err'][i] if 'residual_err' in self.param else 10000
            searcher.FindClosestPoint(pt, self.param)
            cp = self.param['closest'] # update the closest point
            if 'residual_err' in self.param and self.robust:
                # robust pose estimation
                mag_diff = np.linalg.norm(cp-pt)
                mask[i] = mag_diff <= 3 * self.param['err_avg']
            closest_pts.append(cp)
            
        closest_pts = np.asarray(closest_pts).T
        return mask, closest_pts

    def matching(self, points, F):
        """
        compute the closest needlepoint w.r.t. points on surface model 
        """
        searcher = searcher_dict[self.searcher](self.boundingSphere_list, len(self.boundingSphere_list))
        mask, closest_pts = self.search(points, F, searcher)
        return mask, closest_pts



    def __call__(self, points, surf_pts, surf_face):

        self.read_surf(surf_pts, surf_face)
        F = Cartesian_transformation({'R':np.eye(3),'t':np.ones((3,1),np.float32)})
        terminate = True
        while (terminate):
            
            mask, closest_pts = self.matching(points,F)
            F = self.find_best_transform(points, closest_pts, mask)
            criteria = (self.param['err_avg']+1e-6)/(self.param['lst_err_avg']+1e-6)
            terminate = ~((self.param['sigma'] <= self.threshold[0] and\
                        self.param['err_avg'] <= self.threshold[1] and\
                        self.param['err_max'] <= self.threshold[2]) or\
                        (criteria <= 1 and criteria >= 0.95))
        return F, closest_pts




        

