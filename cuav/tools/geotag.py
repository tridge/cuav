#!/usr/bin/python

import numpy, os, time, cv2, sys, math, sys, glob, re
import pyexiv2, datetime, argparse
import fractions, dateutil.parser

from cuav.lib import cuav_util, mav_position


class Fraction(fractions.Fraction):
    """Only create Fractions from floats.

    >>> Fraction(0.3)
    Fraction(3, 10)
    >>> Fraction(1.1)
    Fraction(11, 10)
    """

    def __new__(cls, value, ignore=None):
        """Should be compatible with Python 2.6, though untested."""
        return fractions.Fraction.from_float(value).limit_denominator(99999)

def parse_args():
    '''parse command line arguments'''
    parser = argparse.ArgumentParser("Geotag images from flight log")

    parser.add_argument("files", default=None, help="Image directory or files")
    parser.add_argument("mavlog", default=None, help="flight log for geo-referencing")
    parser.add_argument("--max-deltat", default=0.0, type=float, help="max deltat for interpolation")
    parser.add_argument("--roll-stabilised", default=False, action='store_true', help="Is camera roll stabilised?")
    parser.add_argument("--gps-lag", default=0.0, type=float, help="GPS lag in seconds")
    parser.add_argument("--destdir", default=None, help="destination directory")
    parser.add_argument("--inplace", default=False, action='store_true', help="modify images in-place?")
    return parser.parse_args()


def datetime_to_float(d):
    """Datetime object to seconds since epoch (float)"""
    epoch = datetime.datetime.utcfromtimestamp(0)
    total_seconds =  (d - epoch).total_seconds()
    # total_seconds will be in decimals (millisecond precision)
    return total_seconds


def decimal_to_dms(decimal):
    """Convert decimal degrees into degrees, minutes, seconds.

    >>> decimal_to_dms(50.445891)
    [Fraction(50, 1), Fraction(26, 1), Fraction(113019, 2500)]
    >>> decimal_to_dms(-125.976893)
    [Fraction(125, 1), Fraction(58, 1), Fraction(92037, 2500)]
    """
    remainder, degrees = math.modf(abs(decimal))
    remainder, minutes = math.modf(remainder * 60)
    return [Fraction(n) for n in (degrees, minutes, remainder * 60)]


def set_gps_location(file_name, lat, lng, alt, t):
    """
    see: http://stackoverflow.com/questions/453395/what-is-the-best-way-to-geotag-jpeg-images-with-python
    
    Adds GPS position as EXIF metadata

    Keyword arguments:
    file_name -- image file 
    lat -- latitude (as float)
    lng -- longitude (as float)

    """

    m = pyexiv2.ImageMetadata(file_name)
    m.read()

    m["Exif.GPSInfo.GPSLatitude"] = decimal_to_dms(lat)
    m["Exif.GPSInfo.GPSLatitudeRef"] = 'N' if lat >= 0 else 'S'
    m["Exif.GPSInfo.GPSLongitude"] = decimal_to_dms(lng)
    m["Exif.GPSInfo.GPSLongitudeRef"] = 'E' if lng >= 0 else 'W'
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
  types = ('*.png', '*.jpeg', '*.jpg')
  if os.path.isdir(args.files):
    for tp in types:
        files.extend(glob.glob(os.path.join(args.files, tp)))
  else:
    files.append(args.files)
  files.sort()
  num_files = len(files)
  print("num_files=%u" % num_files)

  mpos = mav_position.MavInterpolator(gps_lag=args.gps_lag)
  mpos.set_logfile(os.path.join(os.getcwd(), args.mavlog))

  frame_time = 0

  if args.destdir:
    cuav_util.mkdir_p(args.destdir)

  for f in files:
    #timestamp is in filename
    timestamp = (os.path.splitext(os.path.basename(f))[0])
    m = re.search("\d", timestamp)
    if m :
        timestamp = timestamp[m.start():]
    
    frame_time = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S%fZ")
    frame_time = datetime_to_float(frame_time)

    try:
      if args.roll_stabilised:
        roll = 0
      else:
        roll = None
      pos = mpos.position(frame_time, args.max_deltat,roll=roll)
    except mav_position.MavInterpolatorException as e:
      print(e)
      pos = None

    if pos:
        im_orig = cv2.imread(f, -1)

        lat_deg = pos.lat
        lng_deg = pos.lon

        if args.inplace:
          basefile = f
        else:
          basefile = os.path.basename(f)
          if args.destdir:
            basefile = os.path.join(args.destdir, os.path.basename(basefile))
        cv2.imwrite(basefile, im_orig)
        count += 1
        
        print("%s %.7f %.7f [%u/%u %.1f%%]" % (os.path.basename(basefile),
                                               lat_deg, lng_deg, count, num_files, (100.0*count)/num_files))
        set_gps_location(basefile, lat_deg, lng_deg, pos.altitude, pos.time)

# main program
if __name__ == '__main__':
    args = parse_args()
    
    process(args)

