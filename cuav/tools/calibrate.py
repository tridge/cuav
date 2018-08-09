#!/usr/bin/env python
"""
This generates a camera calibration matrix (in json form) to correct
for any camera lens distortion.
Requires a set of 10x7 chessboard photos captured via the camera
See ./cuav/data/calibration_images_2015/ChameleonArecort/ for examples
"""

import cv2
import os,sys,string
import argparse
import numpy
from cuav.camera.cam_params import CameraParams


lens=4.0
sensorwidth=5.0

def file_list(directory, extensions):
    '''return file list for a directory'''
    flist = []
    for (root, dirs, files) in os.walk(directory):
        for f in files:
            extension = f.split('.')[-1]
            if extension.lower() in extensions:
                flist.append(os.path.join(root, f))
    return flist
    
def calibrate(imagedir, cbrow, cbcol):
    nimages = 0
    datapoints = []
    im_dims = (0,0)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = numpy.zeros((cbrow * cbcol, 3), numpy.float32)
    objp[:, :2] = numpy.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    files = file_list(imagedir, ['jpg', 'jpeg', 'png'])
    for f in files:
        colour = cv2.imread(f)
        grey = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(grey, (cbcol, cbrow), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
        
        if (ret):
            print('using ' + f)
            cv2.cornerSubPix(grey,corners,(11,11),(-1,-1),(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            objpoints.append(objp)
            imgpoints.append(corners)
            im_dims = grey.shape[:2]

    if len(imgpoints) == 0:
        print("Not enough good quality images. Aborting")
        return
    
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, grey.shape[::-1], None, None)

    # storing results using CameraParams
    C = CameraParams(lens=lens, sensorwidth=sensorwidth, xresolution=im_dims[1], yresolution=im_dims[0])
    C.setParams(K, D)
    C.save(os.path.join(imagedir, "paramsout.json"))
    print("Saved params in " + os.path.join(imagedir, "paramsout.json"))
    

def dewarp(imagedir):
    # Loading from json file
    C = CameraParams.fromfile(os.path.join(imagedir, "params.json"))
    K = C.K
    D = C.D
    print("Loaded camera parameters from " + os.path.join(imagedir, "params.json"))

    for f in file_list(imagedir, ['jpg', 'jpeg', 'png']):
        print(f)
        colour = cv2.imread(f)
        grey = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)

        h, w = grey.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, (w,h), 5)
        dewarped = cv2.remap(grey, mapx, mapy, cv2.INTER_LINEAR)

        x, y, w, h = roi
        dewarped = dewarped[y:y+h, x:x+w]
        grey = cv2.resize(grey, (0,0), fx=0.5, fy=0.5) 
        dewarped = cv2.resize(dewarped, (0,0), fx=0.5, fy=0.5) 

        cv2.imshow("Original", grey )
        cv2.imshow("Dewarped", dewarped)
        cv2.waitKey(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Camera Calibration via chessboard photos")
    parser.add_argument("folder", default=None, help="Image folder or single file")
    parser.add_argument("--chessrow", default=10, type=int, help="Number of rows in the calibration chessboard")
    parser.add_argument("--chesscol", default=7, type=int, help="Number of columns in the calibration chessboard")
    parser.add_argument("--dewarp",dest="dewarp", action='store_true', default=False, help="dewarp gathered images")
    parser.add_argument("--calibrate",dest="calibrate", action='store_true', default=False, help="calculate intrinsics")
    args = parser.parse_args()

    imagedir = args.folder
    if (args.calibrate):
        calibrate(imagedir, args.chessrow, args.chesscol)
    if (args.dewarp):
        dewarp(imagedir)


