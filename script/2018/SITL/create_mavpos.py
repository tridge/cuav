#!/usr/bin/python
'''
create mavpos.dat, containing MavPosition for each image in a set of images using a tlog
'''


import os, time, math, glob, re
import argparse, pickle

from cuav.lib import cuav_util, mav_position


def parse_args():
    '''parse command line arguments'''
    parser = argparse.ArgumentParser("create mavpos.dat")

    parser.add_argument("files", default=None, help="Image directory or files")
    parser.add_argument("mavlog", default=None, help="flight log for geo-referencing")
    parser.add_argument("--roll-stabilised", default=False, action='store_true', help="Is camera roll stabilised?")
    parser.add_argument("--gps-lag", default=0.0, type=float, help="GPS lag in seconds")
    return parser.parse_args()


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

  poshash = {}
    
  for f in files:
    #timestamp is in filename
    timestamp = (os.path.splitext(os.path.basename(f))[0])
    m = re.search("\d", timestamp)
    if m :
        timestamp = timestamp[m.start():]
    
    frame_time = cuav_util.parse_frame_time(f)

    try:
      if args.roll_stabilised:
        roll = 0
      else:
        roll = None
      pos = mpos.position(frame_time, roll=roll)
    except mav_position.MavInterpolatorException as e:
      print(e)
      pos = None

    if pos:
        poshash[os.path.basename(f)] = pos

  dirname = os.path.dirname(f)
  mavpos = os.path.join(dirname, "mavpos.dat")
  open(mavpos,"w").write(pickle.dumps(poshash))
  print("Wrote %s with %u entries" % (mavpos, len(poshash)))

# main program
if __name__ == '__main__':
    args = parse_args()
    
    process(args)

