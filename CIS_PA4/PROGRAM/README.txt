******************************************SOURCE FILES***********************************************
./util/Cartesian_transformation.py                   Cartesian math package
./util/PointCloudRegistration.py                      3D-3D point cloud registration package
./util/FindClosestPoints.py                         Implementation of BrutalSearch, BoundingSphereSearch, KDtreeSearch
./util/Octree.py                                    Implementation of Octree based search
./util/io.py                                        Read and wirte operation
./util/ICP.py                                       Implementation of ICP algorithm.

data/                                               fold containing data files

run.py                                       the whole pipeline to find the registration transformation with ICP algorithm
unit_tes.py                                  unit test for key subroutines, i.e. the robust pose estimation


../OUTPUT/                                           output folder directory for output.txt 
***************************************USING INSTRUCTION******************************************
REQUIREMENTS:
-------------

  This code requires Python 3.8, NumPy, scipy and sklearn,
  To install NumPy, run: pip install numpy 
  To install scipy, run: pip install scipy
  To install sklearn, run: pip install sklearn


How to run our code:
--------------------
  First you need to enter the target directory. For example, in my case: 
  cd F:/Desktop/CISPA2/PROGRAM/

  For running the whole pipeline for one certain case with evaluation (if Output.txt available) and time recording, you need to run:
  python run.py

  To check the options
  >>> python3 run.py -h
  Options:
    -h, --help       show this help message and exit
    --root=ROOT       the folder containing all the data
    --case=CASE       the case ID of data file, e.g. A, B, C, D ...
    -r                use robust estimation
    --method=METHOD   the searching methods including 'Octree',
                      'BoundingSphere', 'BruteForce

Examples
---------
  For running the '-Debug-B-' case with default option (Octree)：
  >>> python run.py --case=B

  For running the '-Unknown-H-' case with default option (Octree)：
  >>> python3 run.py --case=H

  For running the '-Debug-B-' case with various methods:
  >>> python run.py --case=B --method=Octree
  >>> python run.py --case=B --method=BoundingSphere
  >>> python run.py --case=B --method=BrutalSearch

  For running the '-Debug-E-' case with robust pose estimation:
  >>> python run.py --case=E -r


  The returns include the running time and errors(if Answer file given)