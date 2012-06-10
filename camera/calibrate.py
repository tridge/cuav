#!/usr/bin/env python


import cv
import os,sys,string
from numpy import array,zeros

dims=(10,7)
im_w = 640
im_h = 480

def calibrate(imagedir):
  nimages = 0
  datapoints = []
  for f in os.listdir(imagedir):
    image = imagedir+'/'+f
    grey = cv.LoadImage(image,cv.CV_LOAD_IMAGE_GRAYSCALE)
    found,points=cv.FindChessboardCorners(grey,dims,cv.CV_CALIB_CB_ADAPTIVE_THRESH)
    points=cv.FindCornerSubPix(grey,points,(11,11),(-1,-1),(cv.CV_TERMCRIT_EPS+cv.CV_TERMCRIT_ITER,30,0.1))

    if (found):
      print 'using ', image
      nimages += 1
      datapoints.append(points)

  #Number of points in chessboard
  num_pts = dims[0] * dims[1]
  #image points
  ipts = cv.CreateMat(nimages * num_pts, 2, cv.CV_32FC1)
  #object points
  opts = cv.CreateMat(nimages * num_pts, 3, cv.CV_32FC1)
  npts = cv.CreateMat(nimages, 1, cv.CV_32SC1)

  for i in range(0,nimages):
    k=i*num_pts
    for j in range(num_pts):
      cv.Set2D(ipts,k,0,datapoints[i][j][0])
      cv.Set2D(ipts,k,1,datapoints[i][j][1])
      cv.Set2D(opts,k,0,float(j)/float(dims[0]))
      cv.Set2D(opts,k,1,float(j)%float(dims[0]))
      cv.Set2D(opts,k,2,0.0)
      k=k+1
    cv.Set2D(npts,i,0,num_pts)

  K = cv.CreateMat(3, 3, cv.CV_64FC1)
  D = cv.CreateMat(4, 1, cv.CV_64FC1)
  
  cv.SetZero(K)
  cv.SetZero(D)
  
  # focal lengths have 1/1 ratio
  K[0,0] = im_w
  K[1,1] = im_w
  K[0,2] = im_w/2
  K[1,2] = im_h/2
  K[2,2] = 1.0

  rcv = cv.CreateMat(nimages, 3, cv.CV_64FC1)
  tcv = cv.CreateMat(nimages, 3, cv.CV_64FC1)

  #print array(opts)
  #print array(ipts)
  #print array(npts)
  size=cv.GetSize(grey)
  cv.CalibrateCamera2(opts, ipts, npts, size, K, D, rcv, tcv, cv.CV_CALIB_FIX_ASPECT_RATIO|cv.CV_CALIB_USE_INTRINSIC_GUESS|cv.CV_CALIB_ZERO_TANGENT_DIST)

  return K,D

def gather(imagedir):

  c=cv.CaptureFromCAM(0)
  cv.SetCaptureProperty(c,cv.CV_CAP_PROP_FRAME_WIDTH,im_w)
  cv.SetCaptureProperty(c,cv.CV_CAP_PROP_FRAME_HEIGHT,im_h)
  #cv.SetCaptureProperty(c,cv.CV_CAP_PROP_FPS,3.75)
  grey=cv.CreateImage((im_w,im_h),8,1)

  im_cnt = 0
  print 'position chess board then press space to find corners'
  print 'when complete press q'
  if not os.path.exists(imagedir):
    os.mkdir(imagedir)

  if not os.path.exists(imagedir):
    print('failed to create image dir')
    sys.exit()

  while True:
    f=cv.QueryFrame(c)
    if (f == None):
      print 'failed to capture'
      continue
    #print f.width, f.height
    if (f.channels==3):
      cv.CvtColor(f,grey,cv.CV_BGR2GRAY)
    elif (f.channels==1):
      cv.Convert(f,grey)
    else:
      print 'unsupported colourspace'
      break
    key = cv.WaitKey(100)
    key = key & 255
    if not key in range(128):
      # show the image
      cv.ShowImage("calibrate", grey)
      continue
    print 'key=0x%x(\'%c\')' % (key, chr(key))
    key = chr(key)
    if (key == ' '):
      print 'looking for corners...'
      found,points=cv.FindChessboardCorners(grey,dims,cv.CV_CALIB_CB_ADAPTIVE_THRESH)
      if found == 0:
        print 'Failed to find corners. Reposition and try again'
      else:
        cv.DrawChessboardCorners(f,dims,points,found)
        #show the final image
        cv.ShowImage("calibrate",f)
        #wait indefinitely
        print 'Keep this image ? y/n'
        key = chr(cv.WaitKey(0) & 255)
        if (key == 'y'):
          print 'Keeping image ', im_cnt
          cv.SaveImage(imagedir+'/calib%05d.pgm' % (im_cnt), grey )
          im_cnt+=1
        else:
          print 'discarding image'
        print 'press any key to find next corners'
    elif (key == 'q'):
      print 'quit'
      break

if __name__ == '__main__':
  from optparse import OptionParser
  parser = OptionParser("calibrate.py [options]")
  parser.add_option("--gather",dest="gather", action='store_true', default=False, help="gather calibration images")
  parser.add_option("--calibrate",dest="calibrate", action='store_true', default=False, help="calculate intrinsics")
  (opts, args) = parser.parse_args()
  
  if len(args) < 1:
      print("Usage: calibrate.py [options] <imagedir>")
      print("Use --help option for options")
      sys.exit(1)

  imagedir = args[0]
  if (opts.gather):
    gather(imagedir)

  if (opts.calibrate):
    K, D = calibrate(imagedir)
    print array(K)
    print array(D)


