#!/usr/bin/python
'''
convert images from PGM to other formats
'''

import os, sys, glob, cv

from cuav.lib import cuav_util

def parse_args():
  '''parse command line arguments'''
  if 1 == len(sys.argv):
    from MAVProxy.modules.lib.optparse_gui import OptionParser
    file_type='file'
    directory_type='directory'
  else:
    from optparse import OptionParser
    file_type='str'
    directory_type='str'

  parser = OptionParser("pgm_convert.py [options] <directory>")
  parser.add_option("--directory", default=None, type=directory_type,
                    help="directory containing PGM image files")
  parser.add_option("--output-directory", default=None, type=directory_type,
                    help="directory to use for converted files")
  parser.add_option("--format", default='png', help="type of file to convert to (png or jpg)")
  return parser.parse_args()

if __name__ == '__main__':
  (opts, args) = parse_args()

def process(args):
  '''process a set of files'''

  files = []
  for a in args:
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
      if not opts.output_directory:
        outdir = os.path.dirname(f)
      else:
        outdir = opts.output_directory
      basename = os.path.basename(f)[:-4]
      new_name = os.path.join(outdir, basename + '.' + opts.format)
      print("Creating %s" % new_name)
      cv.SaveImage(new_name, im_orig)

if __name__ == '__main__':
  # main program
  if opts.directory is not None:
    process([opts.directory])
  else:
    process(args)
