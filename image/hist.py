#!/usr/bin/env python

from PIL import Image
from numpy import array, reshape, zeros, ones, histogramdd, sqrt, sum, transpose
from matplotlib import pyplot

bsize = 16

if __name__ == '__main__':

  from optparse import OptionParser
  parser = OptionParser("hist.py [options] <filename>")
  (opts, args) = parser.parse_args()

  if len(args) < 2:
    print("Usage: hist.py FILE1 FILE2")
    sys.exit(1)

  ii_file = args[0]
  ri_file = args[1]

  ii = Image.open(ii_file);
  ii_rgb = array(ii.getdata());
  ri = Image.open(ri_file);
  ri_rgb = array(ri.getdata());

  # train background histogram
  (ri_hist, ri_edges) = histogramdd(ri_rgb, bins = (8,8,8), normed=True)

  ii_rgb = reshape(ii_rgb, (ii.size[1], ii.size[0], 3))

  out = zeros((ii.size[1],ii.size[0]));
  for y in range(0, ii.size[1], bsize):
    for x in range(0, ii.size[0], bsize):
      block = ii_rgb[y:y+bsize,x:x+bsize].reshape(bsize*bsize,3);
      (block_hist, block_edges) = histogramdd(block, bins = (8,8,8), normed=True)
      bhutta_dist = sum(sqrt(block_hist.flatten()*ri_hist.flatten()))
      out[y:y+bsize,x:x+bsize] = bhutta_dist * ones((bsize,bsize))

  out /= out.max();

  pyplot.figure(1)
  pyplot.imshow(out)
  pyplot.imsave('p_notjoe.png', out);

  pyplot.figure(2)
  pyplot.imshow(ii_rgb.astype('uint8'))
  ii.save('i_hasjoe.jpg');
  pyplot.show()









