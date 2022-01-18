import numpy as np
import pandas as pd
from util.CalibrateDistortion import DistortionCalib
from util.LoadSensorData import Compute_expectedC
from EMpivotCalibration import EMpivotCalibration, Compute_F
from SolveFreg import SolveFreg
import os
from optparse import OptionParser

class Navigation(object):
    def __init__(self, dewarping=True, eval=True):
        self.dewarping = dewarping
        self.eval = eval
    
    def read(self,root= './data', status ='debug',case='c'):
        self.root = root
        self.case = case
        file = os.path.join(root,'pa2-'+status+'-'+case+'-EM-nav.txt')
        if status == 'debug':
            output = pd.read_csv(file.replace('-EM-nav', '-output2'))
            data = [v[None,:] for v in output.values]
            self.gt = np.concatenate(data, axis=0)
  
        self.status=status
        data = pd.read_csv(file)
        row_num = data.shape[0]
        col_num = data.shape[1]
        heads = [int(c) for c in list(data.columns)[:-1]]
        N_G = heads[0]
        self.N_frame = heads[1]
        s = 0
        e = 6
        frame = []
        for i in range(int(row_num/N_G)):
            frame.append(np.array(data.iloc[s:e].values))
            s = s + 6
            e = e + 6
        for j in range(len(frame)):
            frame[j] = np.transpose(frame[j])
        return frame

    def get_distortion_calibrator(self, root, status, case,order):
        # obtain sensor data {C}, fixed points {c} with respect to local coordinate system, F_D and F_A for each frame
        sensor_data, expected_data = Compute_expectedC(root, status, case, self.eval)
        if self.dewarping == False:
            self.calib = None
            return self.calib, expected_data
        calib_dewrap = DistortionCalib(order)
        calib_dewrap.fit(sensor_data, expected_data)
        self.calib = calib_dewrap
        return self.calib, expected_data

    def distortion_correction(self, frame, calib):
        # distortion correction
        frame_dewrap = []
        for i in range(len(frame)):
            frame_temp = frame[i]
            for j in range(6):
                if self.dewarping:
                    frame_dewrap.append(calib.predict(frame_temp[:,j][:,None]))
                else:
                    frame_dewrap.append(frame_temp[:,j][None,:])
        frame_dewrap = np.concatenate(frame_dewrap, axis = 0)
        return frame_dewrap

    def em_pivot_calibration(self,calib_dewrap):
        # EMpivotCalib
        p_tip, p_dim, point_G_fixed = EMpivotCalibration(self.root, self.status, self.case, self.dewarping, self.eval, calib_dewrap)
        return p_tip, p_dim, point_G_fixed


    def compute_F_t(self, frame, point_G_fixed):
        # compute F_t, the tool coordinate system with respect to the EM tracker system 
        F_t_sequence = []
        s = 0
        e = 6
        for i in range(self.N_frame):
            G_dewrap = np.transpose(frame[s:e, :])
            F_t_sequence.append(Compute_F(point_G_fixed, G_dewrap))
            s = s + 6
            e = e + 6
        return F_t_sequence
    
    def compute_b(self, F_t_sequence, p_tip):
        # compute b
        b = []
        F_reg = SolveFreg(self.root, self.status, self.case, self.dewarping, self.eval, self.calib)
        for i in range(self.N_frame):
            F = F_reg * F_t_sequence[i]
            b.append(F(p_tip[:,None]))
        return np.transpose(np.concatenate(b, axis=1))
    
    def evaluate(self, b):
        errors_frames = []
        l2 = lambda a,b: np.linalg.norm(a-b)
        mae = lambda a,b: np.sum(abs(a-b))# compute the mae loss between two column vectors
        error_frames = [l2(b[i], self.gt[i]) for i in range(self.N_frame) ]
        print('Navigation error of case '+self.case+' : ', error_frames)
        print('Navigation average error of case '+self.case+' : ', np.mean(error_frames))

    def output2_file(self, data, output_root, status, case):
        """
        Output the output2.txt

        Parameters
        ----------
        data : Nx3 ndarray, points in CT coordinate system 
        """
        file_name = 'pa2-' + status + '-' + case +'-output2.txt'
        file_root = os.path.join(output_root, 'pa2-' + status + '-' + case +'-output2.txt')
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        N_b = str(len(data))
        dataframe = pd.DataFrame({N_b:data[:, 0].tolist(),file_name: data[:, 1].tolist(), ' ':data[:, 2].tolist()})
        dataframe.to_csv(file_root, index=False,sep=',')

    def output1_file(self, p_dim,  data_c, output_root, status, case):
        """
        Save the output1.txt. 
        In this assignment, dewarping function is used to improve the accuracy of 
        p_dimple. 
        Since Optical Pivot Calibration is not required, we use 0.00,0.00,0.00 
        to take the place.

        Parameters
        ----------
        p_dim : 3x1 ndarray, dample position with respect to EM tracker
        c_data: 3xN ndarray, expected data
        """
        file_name = 'pa2-' + status + '-' + case +'-output1.txt'
        file_root = os.path.join(output_root, file_name)
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        N_c = str(len(data_c))
        data = np.concatenate([p_dim,np.zeros((3,1)), data_c], axis = 1)
        dataframe = pd.DataFrame({'27':data[0, :].tolist(),'125': data[1, :].tolist(), file_name:data[2,:].tolist()})
        dataframe.to_csv(file_root, index=False,sep=',')


    def __call__(self,root= './data', status ='debug',case='c', output_root = '../OUTPUT', order=5):
        """
        the whole pipeline of PA 2 including distortion calibration, pivot calibration and
        navigation.  

        Parameters
        ----------
        root: the folder containing all the data
        status: 'debug' or 'unknown'
        case:  the case ID of data file, e.g. 'a', 'd', 'f'...
        output_root: the folder path to save the output2.txt and output1.txt

        Return
        ----------
        b: N_frame x 3 ndarray, pointer tip coordinates for each frame relative to CT tracker coordinate system 

        Save
        ----------
        output1.txt (missing the result of optical pivot calibration)
        output2.txt

        """
        self.root = root
        self.status = status
        self.case = case

        # fit the dewarping function by the sensor data {C} and corresponding expected {C_expected}(ground-truth) during calibration
        
        calib_dewrap, data_c = self.get_distortion_calibrator(root, status, case, order)

        # EM-pivot-calibration with dewarping function to obtain accurate p_tip, p_dimple and corresponding {g[0]}
        p_tip, p_dimple, point_G_fixed = self.em_pivot_calibration(calib_dewrap)

        # load the navigation sensor data (without ground-truth) during surgery  
        sensor_data = self.read(root,status,case) 

        # dewarp the sensor data during surgery
        dewrap_sensor_data= self.distortion_correction(sensor_data, calib_dewrap)


        F_t_sequences = self.compute_F_t(dewrap_sensor_data, point_G_fixed)

        b = self.compute_b(F_t_sequences, p_tip)

        if self.status == 'debug':
            self.evaluate(b)

        # save output2.txt
        self.output2_file(b, output_root, status, case)
        # save output1.txt, we ignored the optical pivot calibration by inserting the 0.0,0.0,0.0
        self.output1_file(p_dimple[:,None], data_c, output_root, status, case)
        return b

if __name__=='__main__':

    parser = OptionParser()

    parser.add_option("--root", dest="root", help="the folder containing all the data", default='./data', type="string")
    parser.add_option("--status", dest="status", help="debug or unknown", default='unknown', type="string")
    parser.add_option("--case", dest="case", help=" the case ID of data file, e.g. a, d, f...", default='g', type="string")
    parser.add_option("--order", dest="order", help=" the highest order of polyomial", default=5, type="int")
    parser.add_option("-d", help= " dewarp the distorted sensor data",action="store_true", dest="dewarping", default=True)
    parser.add_option("-c", help= " close the dewarping ", action="store_false", dest="dewarping")
    parser.add_option("-e", help= " evaluate the results of subroutines ", action="store_true", dest="eval")
    parser.add_option("-q", help= " close the evaluation of subroutine results ", action="store_false", dest="eval", default=False)

    (options, args) = parser.parse_args()


    nav = Navigation(options.dewarping, options.eval)
    b = nav(options.root, options.status, options.case, order = options.order)
    for i in range(len(b)):
        print('The b coordinates for Frame {}:  '.format(i), b[i])

