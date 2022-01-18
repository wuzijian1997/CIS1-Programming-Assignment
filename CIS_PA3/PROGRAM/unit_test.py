import numpy as np
import pandas as pd
import os
from optparse import OptionParser
from util.io import read_output, read_surface, result_evaluation, write_output, read_markers_to_body, read_markers_to_tracker
from util.PointCloudRegistration import Point2Point_Reg
from sklearn.neighbors import KDTree
import time
from util.FindClosestPoints import ClosestTriangleSearch

## test ComputeBoudingSphere(vertex)
verter_tri = [[ 0,  0,  0],
              [ 1,  1,  0],
              [ 2,  0,  0]]
center, radius = ClosestTriangleSearch().ComputeBoudingSphere(verter_tri)
print("center is ", center)
print("radius = ", radius)

## test ComputeClosestPoint(point, vertex)
verter = [[-1,  0,  0],
          [ 1,  1,  0],
          [ 1, -1,  0]]
# case 1, c is in the vertex of the triangle
point_1 = [-1, 0, 1]
h = ClosestTriangleSearch().ComputeClosestPoint(point_1, verter)
print("closest point is ", h)

# case 2, c is in the side of the triangle
point_2 = [1, 0, 1]
h = ClosestTriangleSearch().ComputeClosestPoint(point_2, verter)
print("closest point is ", h)

# case 3, c is in the exterior of the triangle
point_3 = [2, 0, 1]
h = ClosestTriangleSearch().ComputeClosestPoint(point_3, verter)
print("closest point is ", h)

# case 4, c is in the inside of the triangle
point_4 = [0, 0, 1]
h = ClosestTriangleSearch().ComputeClosestPoint(point_4, verter)
print("closest point is ", h)