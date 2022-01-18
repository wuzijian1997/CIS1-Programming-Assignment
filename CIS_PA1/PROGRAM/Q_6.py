from numpy.lib.function_base import append
from pandas.core import frame
from util.Cartesian_transformation import Cartesian_transformation
from util.PointCloudRegistration import Point2Point_Reg
from util.pivot_calib import Pivot_calib
import numpy as np
import pandas as pd

# compute transformation
# active: FAB = Compute_F(A, B)
# passive: FAB = Compute_F(B, A)
def Compute_F(points, Points):
    Reg = Point2Point_Reg()
    return Reg(points, Points)


data = pd.read_csv("./data/pa1-unknown-k-optpivot.txt")
heads = [int(c) for c in list(data.columns)[:-1]]
N_D = heads[0] # number of marker on EMbase
N_H = heads[1] # number of marker on tool
N_frame = heads[2] # number of frame
data.columns = ['H_D', 'N_H', 'N_frame', 'NAN']
data = data.dropna(how='any', axis=1)
#print(data)

# get d
d = np.asarray([[0.00,     0.00,   0.00],
                [0.00,     0.00,   150.00],
                [0.00,     150.00, 0.00],
                [0.00,     150.00, 150.00],
                [150.00,   0.00,   0.00],
                [150.00,   0.00,   150.00],
                [150.00,   150.00, 0.00],
                [150.00,   150.00, 150.00]])
#print(d)

# split data as frames
s = 0
e = N_D + N_H
frame_D = []
frame_H = []
for i in range(int(N_frame)):
    frame_temp = np.array(data.iloc[s:e].values)
    frame_D.append(frame_temp[0 : N_D])
    frame_H.append(frame_temp[N_D : (N_D + N_H)])
    s = s + N_D + N_H
    e = e + N_D + N_H
#print(frame_D)
#print(frame_H)

# compute F_D
FD_inv = []
for i in range(int(N_frame)):
    FD_inv.append(Compute_F(np.transpose(d), np.transpose(frame_D[i])).inverse())

# transfer H from opt to EM
H_EM = []
for i in range(int(N_frame)):
    H_EM.append(FD_inv[i](np.transpose(frame_H[i]))) # np.matmul(FD_inv.param['R'], np.transpose(frame_H[i])) + FD_inv.param['t'])
#print(H_EM)

# compute H_0
H_1 = H_EM[0]
H_0 = np.mean(H_1, axis = 1, keepdims = True)
F_1 = Cartesian_transformation({'R':np.eye(3), 't':H_0})
#print(F_1.param)
#print(H_0)

# compute h_i
h = []
for i in range(int(N_frame)):
    h.append(H_EM[i] - np.mean(H_EM[i], axis=1, keepdims=True))
#print(h[0])

# compute F_H[k]
F_H = []
for i in range(int(N_frame)):
    print()
    F = Compute_F(h[0], H_EM[i])
    # print(F.param)
    # F = Compute_F(np.transpose(g[i]), np.transpose(g[0]))
    if np.sum(F.param['R']) != 0:
        #F.param['t'] = np.transpose(np.mean(frame[i], axis=0, keepdims=True))
        F_H.append(F)
        #print(F.param)

# compute p_dimple
calib = Pivot_calib()
p_dimple = calib(F_H)
print('p_dimple is :    ',p_dimple)