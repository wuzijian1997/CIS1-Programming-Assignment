******************************************SOURCE FILES***********************************************
./util/Cartesian_transformation.py                   Cartesian math package
./util/PointCloudRegistration.py                      3D-3D point cloud registration package
./util/FindClosestPoints.py                         Implementation of BrutalSearch, BoundingSphereSearch, KDtreeSearch
./util/Octree.py                                    Implementation of Octree based search
./util/io.py                                        Read and wirte operation

data/                                               fold containing data files

run.py                                       the whole pipeline to find the paired points on surface model
unit_tes.py                                  unit test for key subroutines, i.e. BoudingSphere and closest point on triangle detection 


../OUTPUT/                                           output folder directory for output1.txt and output2.txt 
***************************************USING INSTRUCTION******************************************
REQUIREMENTS:

This code requires Python 3.8, NumPy, scipy and sklearn,
To install NumPy, run: pip install numpy 
To install scipy, run: pip install scipy
To install sklearn, run: pip install sklearn


How to run our code:
First you need to enter the target directory. For example, in my case: 
cd F:/Desktop/CISPA2/PROGRAM/

For running the whole pipeline for one certain case with evaluation (if Output.txt available) and time recording, you need to run:
python run.py

To check the options
>>> python3 run.py -h
Options:
  -h, --help   show this help message and exit
  --root=ROOT  the folder containing all the data
  --case=CASE   the case ID of data file, e.g. A, B, C, D ...

Examples. 
For running the '-Debug-B-' case：
>>> python3 run.py --case=B

For running the '-Unknown-H-' case：
>>> python3 run.py --case=H

