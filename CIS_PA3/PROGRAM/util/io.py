import pandas as pd
import os
import numpy as np

def read_markers_to_body(root):
    content = pd.read_csv(root)
    head = content.columns[0]
    marker_num = int(head.split(' ')[0])
    values = content.values
    assert(len(values) == marker_num + 1)
    data = []
    for i in range(marker_num+1):
        line = values[i]
        assert len(line) == 1
        line = [float(c) for c in line[0].split(' ') if c != '']
        assert len(line) == 3
        data.append(line)
    # print(values)
    # print(data)
    data = np.transpose(np.asarray(data,np.float32))
    return data[:, :marker_num], data[:, marker_num:]

def read_markers_to_tracker(root, num_marker = 6):
    """
    Return
    ----------
    A,B: 3d array with the shape of (N_Frame x 3 x  marker_num)
    """
    content = pd.read_csv(root)
    heads= [int(col) for col in content.columns[:-1]]
    N_s = heads[0]
    N_D = N_s - num_marker * 2
    N_Frame = heads[1]
    data = np.asarray(content.values, np.float32)
    data = np.transpose(data)
    A = [data[:,N_s*i: N_s*i + num_marker] for i in range(N_Frame)]
    B = [data[:,N_s*i + num_marker: N_s*i + 2*num_marker] for i in range(N_Frame)]
    A = np.asarray(A)
    B = np.asarray(B)
    return A, B

def read_surface(root):
    """
    Return
    ----------
    points: ndarray with the shape of (3 x  point_num)
    faces: ndarray with the shape of (6 x  triangle_num)
    """
    content = pd.read_csv(root)
    points_num = int(content.columns[0])
    values = np.asarray(content.values)
    values = [v[0].split(' ') for v in content.values]
    points = [[float(v[0]), float(v[1]), float(v[2])] for v in values if len(v) == 3]
    faces_num = [int(v[0]) for v in values if len(v) == 1 ]
    faces = [[int(idx) for idx in v] for v in values if len(v) == 6]
    assert len(faces) == faces_num[0] and len(points) == points_num

    points, faces = np.transpose(np.asarray(points)), np.transpose(np.asarray(faces))
    
    return points, faces

def read_output(root):
    content = pd.read_csv(root)
    heads = content.columns
    N_Frame = int(heads[0].split(' ')[0])
    data = []
    values = content.values
    for i in range(N_Frame):
        line = values[i]
        assert len(line) == 1
        line = [float(c) for c in line[0].split(' ') if c != '']
        assert len(line) == 7
        data.append(line)  
    data = np.transpose(np.asarray(data, np.float32))
    mags = data[-1, :]
    d_gt = data[:3, :]
    c_gt = data[3:-1, :] 
    return d_gt, c_gt, mags

def write_output(root, data, N_frame=15):
    root_dir = root.split('PA3-')[0]
    if not os.path.exists(root_dir):
            os.makedirs(root_dir)
    head_str = '  '.join([str(N_frame), root.split('/')[-1]])
    if len(data) == 7:
        data = np.transpose(data)
    values = []
    for i in range(len(data)):
        line = data[i]
        line = [str(line[j]) for j in range(len(line))]
        values.append('   '.join(line))
    dataframe = pd.DataFrame({head_str: values})
    dataframe.to_csv(root, index=False,sep=' ')

def result_evaluation(points_debug, points_result):
    """
    Use mean L-2 norm to evaluate the result
    """
    sum = 0
    length = len(points_debug) if len(points_debug.shape) == 1 else points_debug.shape[1]
    for i in range(length):
        points_debug_i = points_debug[:,i] if len(points_debug.shape) == 2 else points_debug[i]
        points_result_i = points_result[:,i] if len(points_result.shape) == 2 else points_result[i]
        sum = sum + np.linalg.norm(points_debug_i- points_result_i)
    L2_mean = sum / length

    return L2_mean

if __name__ == '__main__':
    root = './data/PA3-B-Debug-Output.txt'
    output_root = './output/PA3-B-Debug-Output.txt'
    d_gt, c_gt, mags = read_output(root)
    points, faces = read_surface('./data/Problem3Mesh.sur')
    print(points.shape)
    print(faces)