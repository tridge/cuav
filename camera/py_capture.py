#!/usr/bin/python

from numpy import array,zeros
from matplotlib import pyplot

import chameleon

colour = 0
depth = 8
h = chameleon.open(colour, depth)

im = zeros((960,1280),dtype='uint8')
f = pyplot.figure(1)

for i in range(0,10):
  try:
    sts = chameleon.trigger(h)
    if (sts == 0):
      (shutter, ftime) = chameleon.capture(h, im)
  except chameleon.error:
    print('failed to capture')

pyplot.imshow(im)
f.show()
chameleon.close(h)

k = raw_input()

