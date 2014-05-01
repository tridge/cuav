#!/usr/bin/python

import numpy, os, time, cv, sys, math, sys, glob
import pyexiv2, datetime

from cuav.lib import cuav_util, cuav_mosaic, mav_position, cuav_joe, cuav_region
from MAVProxy.modules.mavproxy_map import mp_slipmap
from MAVProxy.modules.lib import mp_image

from optparse import OptionParser
parser = OptionParser("geotag.py [options] <directory|files>")
parser.add_option("--mavlog", default=None, help="flight log for geo-referencing")
parser.add_option("--max-deltat", default=0.0, type='float', help="max deltat for interpolation")
parser.add_option("--max-attitude", default=45, type='float', help="max attitude geo-referencing")
parser.add_option("--lens", default=4.0, type='float', help="lens focal length")
parser.add_option("--roll-stabilised", default=False, action='store_true', help="roll is stabilised")
parser.add_option("--gps-lag", default=0.0, type='float', help="GPS lag in seconds")
parser.add_option("--destdir", default=None, help="destination directory")
parser.add_option("--inplace", default=False, action='store_true', help="in-place modify")
(opts, args) = parser.parse_args()

def to_deg(value, loc):
  if value < 0:
    loc_value = loc[0]
  elif value > 0:
    loc_value = loc[1]
  else:
    loc_value = ""
  abs_value = abs(value)
  deg =  int(abs_value)
  t1 = (abs_value-deg)*60
  min = int(t1)
  sec = round((t1 - min)* 60, 5)
  return (deg, min, sec, loc_value)
      
def set_gps_location(file_name, lat, lng, alt, t):
    """
    see: http://stackoverflow.com/questions/453395/what-is-the-best-way-to-geotag-jpeg-images-with-python
    
    Adds GPS position as EXIF metadata

    Keyword arguments:
    file_name -- image file 
    lat -- latitude (as float)
    lng -- longitude (as float)

    """

    lat_deg = to_deg(lat, ["S", "N"])
    lng_deg = to_deg(lng, ["W", "E"])

    # convert decimal coordinates into degrees, munutes and seconds
    exiv_lat = (pyexiv2.Rational(lat_deg[0]*60+lat_deg[1],60),
                pyexiv2.Rational(lat_deg[2]*100,6000),
                pyexiv2.Rational(0, 1))
    exiv_lng = (pyexiv2.Rational(lng_deg[0]*60+lng_deg[1],60),
                pyexiv2.Rational(lng_deg[2]*100,6000),
                pyexiv2.Rational(0, 1))

    m = pyexiv2.ImageMetadata(file_name)
    m.read()

    m["Exif.GPSInfo.GPSLatitude"] = exiv_lat
    m["Exif.GPSInfo.GPSLatitudeRef"] = lat_deg[3]
    m["Exif.GPSInfo.GPSLongitude"] = exiv_lng
    m["Exif.GPSInfo.GPSLongitudeRef"] = lng_deg[3]
    m["Exif.Image.GPSTag"] = 654
    m["Exif.GPSInfo.GPSMapDatum"] = "WGS-84"
    m["Exif.GPSInfo.GPSVersionID"] = '2 0 0 0'
    m["Exif.Image.DateTime"] = datetime.datetime.fromtimestamp(t)

    try:
      m["Exif.GPSInfo.GPSAltitude"] = mav_position.Fraction(alt)
    except Exception:
      pass

    m.write()
    

def process(args):
  '''process a set of files'''

  count = 0
  files = []
  for a in args:
    if os.path.isdir(a):
      files.extend(glob.glob(os.path.join(a, '*.png')))
    else:
      files.append(a)
  files.sort()
  num_files = len(files)
  print("num_files=%u" % num_files)

  if opts.mavlog:
    mpos = mav_position.MavInterpolator(gps_lag=opts.gps_lag)
    mpos.set_logfile(opts.mavlog)
  else:
    print("You must provide a mavlink log file")
    sys.exit(1)

  frame_time = 0

  if opts.destdir:
    cuav_util.mkdir_p(opts.destdir)

  for f in files:
    frame_time = os.path.getmtime(f)
    try:
      if opts.roll_stabilised:
        roll = 0
      else:
        roll = None
      pos = mpos.position(frame_time, opts.max_deltat,roll=roll)
    except mav_position.MavInterpolatorException as e:
      print e
      pos = None

    im_orig = cv.LoadImage(f)

    lat_deg = pos.lat
    lng_deg = pos.lon

    if opts.inplace:
      newfile = f
    else:
      basefile = f.split('.')[0]
      newfile = basefile + '.jpg'    
      if opts.destdir:
        newfile = os.path.join(opts.destdir, os.path.basename(newfile))
    cv.SaveImage(newfile, im_orig)
    count += 1
    
    print("%s %.7f %.7f [%u/%u %.1f%%]" % (os.path.basename(newfile),
                                           lat_deg, lng_deg, count, num_files, (100.0*count)/num_files))
    set_gps_location(newfile, lat_deg, lng_deg, pos.altitude, pos.time)

# main program

process(args)
