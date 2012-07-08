#!/usr/bin/env python


import cv
import os,sys,string
from numpy import array,zeros
from camera import CameraParams

dims=(10,7)

def calibrate(imagedir):
  nimages = 0
  datapoints = []
  im_dims = (0,0)
  for f in os.listdir(imagedir):
    if (f.find('pgm')<0):
      continue
    image = imagedir+'/'+f
    grey = cv.LoadImage(image,cv.CV_LOAD_IMAGE_GRAYSCALE)
    found,points=cv.FindChessboardCorners(grey,dims,cv.CV_CALIB_CB_ADAPTIVE_THRESH)
    points=cv.FindCornerSubPix(grey,points,(11,11),(-1,-1),(cv.CV_TERMCRIT_EPS+cv.CV_TERMCRIT_ITER,30,0.1))

    if (found):
      print 'using ', image
      nimages += 1
      datapoints.append(points)
      im_dims = (grey.width, grey.height)

  #Number of points in chessboard
  num_pts = dims[0] * dims[1]
  #image points
  ipts = cv.CreateMat(nimages * num_pts, 2, cv.CV_32FC1)
  #object points
  opts = cv.CreateMat(nimages * num_pts, 3, cv.CV_32FC1)
  npts = cv.CreateMat(nimages, 1, cv.CV_32SC1)

  for i in range(0,nimages):
    k=i*num_pts
    squareSize = 1.0
    # squareSize is 1.0 (i.e. units of checkerboard)
    for j in range(num_pts):
      cv.Set2D(ipts,k,0,datapoints[i][j][0])
      cv.Set2D(ipts,k,1,datapoints[i][j][1])
      cv.Set2D(opts,k,0,float(j%dims[0])*squareSize)
      cv.Set2D(opts,k,1,float(j/dims[0])*squareSize)
      cv.Set2D(opts,k,2,0.0)
      k=k+1
    cv.Set2D(npts,i,0,num_pts)

  K = cv.CreateMat(3, 3, cv.CV_64FC1)
  D = cv.CreateMat(5, 1, cv.CV_64FC1)
  
  cv.SetZero(K)
  cv.SetZero(D)
  
  # focal lengths have 1/1 ratio
  K[0,0] = im_dims[0]
  K[1,1] = im_dims[0]
  K[0,2] = im_dims[0]/2
  K[1,2] = im_dims[1]/2
  K[2,2] = 1.0

  rcv = cv.CreateMat(nimages, 3, cv.CV_64FC1)
  tcv = cv.CreateMat(nimages, 3, cv.CV_64FC1)

  #print 'object'
  #print array(opts)
  #print 'image'
  #print array(ipts)
  #print 'npts'
  #print array(npts)

  size=cv.GetSize(grey)
  flags = 0
  #flags |= cv.CV_CALIB_FIX_ASPECT_RATIO
  #flags |= cv.CV_CALIB_USE_INTRINSIC_GUESS
  #flags |= cv.CV_CALIB_ZERO_TANGENT_DIST
  #flags |= cv.CV_CALIB_FIX_PRINCIPAL_POINT
  cv.CalibrateCamera2(opts, ipts, npts, size, K, D, rcv, tcv, flags)

  # storing results using CameraParams
  C = CameraParams(xresolution=im_dims[0], yresolution=im_dims[1])
  print array(K)
  print array(D)
  C.setParams(K, D)
  C.save(imagedir+"/params.json")

def dewarp(imagedir):
  # Loading from json file
  C = CameraParams()
  C.load(imagedir+"/params.json")
  K = cv.fromarray(C.K)
  D = cv.fromarray(C.D)
  print "loaded camera parameters"
  mapx = None
  mapy = None
  for f in os.listdir(imagedir):
    if (f.find('pgm')<0):
      continue
    image = imagedir+'/'+f
    print image
    original = cv.LoadImage(image,cv.CV_LOAD_IMAGE_GRAYSCALE)
    dewarped = cv.CloneImage(original);
    # setup undistort map for first time
    if (mapx == None or mapy == None):
      im_dims = (original.width, original.height)
      mapx = cv.CreateImage( im_dims, cv.IPL_DEPTH_32F, 1 );
      mapy = cv.CreateImage( im_dims, cv.IPL_DEPTH_32F, 1 );
      cv.InitUndistortMap(K,D,mapx,mapy)

    cv.Remap( original, dewarped, mapx, mapy )

    tmp1=cv.CreateImage((im_dims[0]/2,im_dims[1]/2),8,1)
    cv.Resize(original,tmp1)
    tmp2=cv.CreateImage((im_dims[0]/2,im_dims[1]/2),8,1)
    cv.Resize(dewarped,tmp2)

    cv.ShowImage("Original", tmp1 )
    cv.ShowImage("Dewarped", tmp2)
    cv.WaitKey(-1)


def gather(imagedir, debayer, im_w, im_h):
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
    if (debayer):
      bgr=cv.CreateImage((im_w,im_h),8,3)
      cv.CvtColor(f,bgr,cv.CV_BayerGR2BGR)
      f = bgr

    if (f.channels==3):
      cv.CvtColor(f,grey,cv.CV_BGR2GRAY)
    elif (f.channels==1):
      # convert to 8 bit pixel depth
      cv.Convert(f,grey)
    else:
      print 'unsupported colourspace'
      break
    key = cv.WaitKey(100)
    key = key & 255
    if not key in range(128):
      # show the image
      if (im_w > 640):
        tmp=cv.CreateImage((im_w/2,im_h/2),8,f.channels)
        cv.Resize(f,tmp)
        cv.ShowImage("calibrate", tmp)
      else:
        cv.ShowImage("calibrate", f)
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
        if (im_w > 640):
          tmp=cv.CreateImage((im_w/2,im_h/2),8,f.channels)
          cv.Resize(f,tmp)
          cv.ShowImage("calibrate", tmp)
        else:
          cv.ShowImage("calibrate", f)
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
  parser = OptionParser("calibrate.py [options] <imagedir>")
  parser.add_option("--gather",dest="gather", action='store_true', default=False, help="gather calibration images")
  parser.add_option("--width",dest="width", type='int', default=1280, help="capture width")
  parser.add_option("--height",dest="height", type='int', default=960, help="capture height")
  parser.add_option("--debayer",dest="debayer", action='store_true', default=False, help="debayer input images first")
  parser.add_option("--dewarp",dest="dewarp", action='store_true', default=False, help="dewarp gathered images")
  parser.add_option("--calibrate",dest="calibrate", action='store_true', default=False, help="calculate intrinsics")
  (opts, args) = parser.parse_args()

  if len(args) < 1:
      print("Usage: calibrate.py [options] <imagedir>")
      print("Use --help option for options")
      sys.exit(1)

  imagedir = args[0]
  if (opts.gather):
    gather(imagedir, opts.debayer, opts.width, opts.height)

  if (opts.calibrate):
    calibrate(imagedir)

  if (opts.dewarp):
    dewarp(imagedir)


