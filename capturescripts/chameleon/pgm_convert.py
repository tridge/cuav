#!/usr/bin/python
'''
convert images from PGM to other formats
'''

import os, sys, glob, cv2, argparse

def parse_args():
  '''parse command line arguments'''
  parser = argparse.ArgumentParser("Convert pgm image to png or jpg")
    
  parser.add_argument("directory", default=None,
                    help="directory containing PGM image files")
  parser.add_argument("--output-directory", default=None,
                    help="directory to use for converted files")
  parser.add_argument("--format", default='png', choices=['png', 'jpg'], help="type of file to convert to (png or jpg)")
  return parser.parse_args()

def process(args):
  '''process a set of files'''

  files = []
  if os.path.isdir(args.directory):
    files.extend(glob.glob(os.path.join(args.directory, '*.pgm')))
  else:
    if args.directory.find('*') != -1:
      files.extend(glob.glob(args.directory))
    else:
      files.append(args.directory)
  files.sort()

  for f in files:
      im_orig = cv2.imread(f,-1)
      im_colour = cv2.cvtColor(im_orig, cv2.COLOR_BAYER_GR2BGR)
      if not args.output_directory:
        outdir = os.path.dirname(f)
      else:
        outdir = args.output_directory
      basename = os.path.basename(f)[:-4]
      new_name = os.path.join(outdir, basename + '.' + args.format)
      print("Creating %s" % new_name)
      cv2.imwrite(new_name, im_colour)

if __name__ == '__main__':
  args = parse_args()
    
  # main program
  process(args)
