#!/usr/bin/python
'''
convert images from PGM to other formats
'''

import os, sys, glob, cv, argparse

from cuav.lib import cuav_util

def parse_args():
  '''parse command line arguments'''

  parser = argparse.ArgumentParser("pgm_convert.py")
  parser.add_argument("directory", default=None, nargs='+',
                    help="directory containing PGM image files")
  parser.add_argument("--output-directory", default=None,
                    help="directory to use for converted files")
  parser.add_argument("--format", default='png', help="type of file to convert to (png or jpg)")
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()

def process(args):
  '''process a set of files'''

  files = []
  for a in args.directory:
    if os.path.isdir(a):
      files.extend(glob.glob(os.path.join(a, '*.pgm')))
    else:
      if a.find('*') != -1:
        files.extend(glob.glob(a))
      else:
        files.append(a)
  files.sort()

  for f in files:
      im_orig = cuav_util.LoadImage(f)
      if not args.output_directory:
        outdir = os.path.dirname(f)
      else:
        outdir = args.output_directory
      basename = os.path.basename(f)[:-4]
      new_name = os.path.join(outdir, basename + '.' + args.format)
      print("Creating %s" % new_name)
      cv.SaveImage(new_name, im_orig)

if __name__ == '__main__':
  # main program
  process(args)
