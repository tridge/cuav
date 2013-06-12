#!/usr/bin/python

import numpy, os, time, cv, sys, math, sys, glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'image'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'camera'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'MAVProxy'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'MAVProxy', 'modules'))
import scanner, cuav_util, cuav_mosaic, mav_position, chameleon, cuav_joe, cuav_region, cam_params
from mavproxy_map import mp_image

from optparse import OptionParser
parser = OptionParser("geoshow.py [options] <directory>")
(opts, args) = parser.parse_args()

def process(args):
  '''process a set of files'''

  files = []
  for a in args:
    if os.path.isdir(a):
      files.extend(glob.glob(os.path.join(a, '*.jpg')))
    else:
      files.append(a)
  files.sort()
  num_files = len(files)
  print("num_files=%u" % num_files)

  for f in files:
      pos = mav_position.exif_position(f)
      print pos, time.ctime(pos.time)

# main program

process(args)
