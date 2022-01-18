******************************************SOURCE FILES***********************************************
../util/Cartesian_transformation.py                   Cartesian math package
../util/PointCloudRegistration.py                      3D-3D point cloud registration package
../util/pivot_calib.py                                          "pivot" calibration package 
Q_4.py                                                              program for question 4
Q_5.py                                                              program for question 5 
Q_6.py                                                              program for question 6 w/o evaluation
Q_6_eval.py                                                         program for evaluating the results of question 6

***************************************USING INSTRUCTION******************************************
REQUIREMENTS:

This code requires Python 3.8 and NumPy,
To install NumPy, run: pip install numpy 

How to run our code:
First you need to enter the target directory. For example, in my case: 
cd F:/Desktop/CISPA1/PROGRAM/

For running excutable program, for example, to run Q_4.py for 'debug' file: 
>>> python Q_4.py
Error for each frame:          [0.42232755, 0.47090128, 0.50272006, 0.43490925, 0.47823086, 0.4676946, 0.49983668, 0.49159223]
Average error for all frames:   0.47102657

To run for 'unknown' file, for example, run Q_5.py:
>>> python Q_5.py
p_dimple:    [191.26117329 190.14995776 210.17920719]

