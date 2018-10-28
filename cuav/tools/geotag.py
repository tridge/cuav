#!/usr/bin/python

import os, time, math, glob, sys, re
import piexif, datetime, argparse
import dateutil.parser, shutil
from gooey import Gooey, GooeyParser

from cuav.lib import cuav_util, mav_position

@Gooey
def parse_args_gooey():
    '''parse command line arguments (GUI)'''
    parser = GooeyParser(description="Geotag (EXIF) images from flight log")    

    parser.add_argument("files", default=None, help="Image folder (jpegs only)", widget='DirChooser')
    parser.add_argument("mavlog", default=None, help="flight log for geo-referencing", widget='FileChooser')
    parser.add_argument("--max-deltat", default=0.0, type=float, help="max deltat for interpolation")
    parser.add_argument("--roll-stabilised", default=False, action='store_true', help="Is camera roll stabilised?")
    parser.add_argument("--gps-lag", default=0.0, type=float, help="GPS lag in seconds")
    parser.add_argument("--destdir", default=None, help="destination directory", widget='DirChooser')
    parser.add_argument("--inplace", default=False, action='store_true', help="modify images in-place?")
    return parser.parse_args()

    
def parse_args():
    '''parse command line arguments'''
    parser = argparse.ArgumentParser("Geotag images from flight log")

    parser.add_argument("files", default=None, help="Image directory or files (jpegs only)")
    parser.add_argument("mavlog", default=None, help="flight log for geo-referencing")
    parser.add_argument("--max-deltat", default=0.0, type=float, help="max deltat for interpolation")
    parser.add_argument("--roll-stabilised", default=False, action='store_true', help="Is camera roll stabilised?")
    parser.add_argument("--gps-lag", default=0.0, type=float, help="GPS lag in seconds")
    parser.add_argument("--destdir", default=None, help="destination directory")
    parser.add_argument("--inplace", default=False, action='store_true', help="modify images in-place?")
    return parser.parse_args()


def set_gps_location(file_name, lat, lng, alt, t):
    """
    Adds GPS position as EXIF metadata

    Keyword arguments:
    file_name -- image file 
    lat -- latitude (as float)
    lng -- longitude (as float)

    """
    image_ifd = {piexif.ImageIFD.GPSTag: 654,
                piexif.ImageIFD.DateTime: datetime.datetime.fromtimestamp(t),
                }
    gps_ifd = {piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
           piexif.GPSIFD.GPSAltitudeRef: 0,
           piexif.GPSIFD.GPSLatitude: mav_position.decimal_to_dms(lat),
           piexif.GPSIFD.GPSLatitudeRef: 'N' if lat >= 0 else 'S',
           piexif.GPSIFD.GPSLongitude: mav_position.decimal_to_dms(lng),
           piexif.GPSIFD.GPSLongitudeRef: 'E' if lng >= 0 else 'W',
           piexif.GPSIFD.GPSMapDatum: "WGS-84",
           piexif.GPSIFD.GPSAltitude: (mav_position.Fraction(alt).numerator, mav_position.Fraction(alt).denominator), #Note that this is alt above ground, NOT AMSL
           }
    exif_dict = {"Image":image_ifd, "GPS":gps_ifd}

    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, file_name)


def process(args):
    '''process a set of files'''
    count = 0
    files = []
    types = ('*.jpeg', '*.jpg')
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
        frame_time = cuav_util.datetime_to_float(frame_time)

        try:
            if args.roll_stabilised:
                roll = 0
            else:
                roll = None
            pos = mpos.position(frame_time, args.max_deltat,roll=roll)
        except mav_position.MavInterpolatorException as e:
            print("{0} - {1} ".format(os.path.basename(f), e))
            count += 1
            pos = None

        if pos:
            lat_deg = pos.lat
            lng_deg = pos.lon

            if args.inplace:
                outfile = f
            else:
                basefile = os.path.basename(f)
                outfile = ""
                if args.destdir:
                    outfile = os.path.join(args.destdir, basefile)
                else:
                    outfile = os.path.join(os.getcwd(), basefile)
                shutil.copy2(f, outfile)
            count += 1
            
            print("%s %.7f %.7f [%u/%u %.1f%%]" % (os.path.basename(outfile),
                                                   lat_deg, lng_deg, count, num_files, (100.0*count)/num_files))
            set_gps_location(outfile, lat_deg, lng_deg, pos.altitude, pos.time)
  
# main program
if __name__ == '__main__':
    if not len(sys.argv) > 1:
        args = parse_args_gooey()
    else:
        args = parse_args()
    
    process(args)

