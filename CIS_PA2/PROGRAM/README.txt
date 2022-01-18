******************************************SOURCE FILES***********************************************
./util/Cartesian_transformation.py                   Cartesian math package
./util/PointCloudRegistration.py                      3D-3D point cloud registration package
./util/pivot_calib.py                                          "pivot" calibration package 
./util/LoadSensorData.py                               Compute {C_expected}  and read the sensor data {C}
./util/CalibrateDistortion.py                          Distortion calibration package
data/                                               fold containing data files

Navigation.py                                       the whole pipeline saving output1.txt and output2.txt, including all the subroutins, i.e. calibration and navigation
EMpivotCalibration.py                               program for em pivot calibration 
SolveFreg.py                                        program for solving F_reg
unitTest.py                                         program for verifying the distortion calibration with simulated dataset

../OUTPUT/                                           output folder directory for output1.txt and output2.txt 
***************************************USING INSTRUCTION******************************************
REQUIREMENTS:

This code requires Python 3.8, NumPy and scipy,
To install NumPy, run: pip install numpy 
To install scipy, run: pip install scipy

How to run our code:
First you need to enter the target directory. For example, in my case: 
cd F:/Desktop/CISPA2/PROGRAM/

For running the whole pipeline (calibration and navigation), run Navigation.py with options:
To check the options
>>> python3 Navigation.py -h
    Options:
    -h, --help       show this help message and exit
    --root=ROOT      the folder containing all the data
    --status=STATUS  debug or unknown
    --case=CASE       the case ID of data file, e.g. a, d, f...
    --order=ORDER     the highest order of polyomial, default is 5
    -d                dewarp the distorted sensor data [Default]
    -c                close the dewarping
    -e                evaluate the results of subroutines
    -q                close the evaluation of subroutine results [Default]

Examples. 
For running the -debug-e- case：
>>> python3 Navigation.py --case=e --status=debug

For running the -unknown-h- case：
>>> python3 Navigation.py --case=g --status=unknown

setting the highest order as 3, with dewarping function closed and evaluation of subroutins:
>>> python3 Navigation.py --case=e --status=debug --order=3 -c -e 

Similarly, depends on the version of python you use. 
>>> python Navigation.py --case=e --status=debug --order=4 -e

