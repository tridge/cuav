#!/usr/bin/python

import chameleon, numpy

colour = 1
depth = 8
h = chameleon.open(colour, depth)

im = numpy.zeros((960,1280),dtype='uint8')

for i in range(0,10):
  try:
    chameleon.trigger(h)
    (shutter, ftime) = chameleon.capture(h, im)
    filename = 'i%u.pgm' % i
    chameleon.save_pgm(h, filename, im)
    print("Captured to %s" % filename)
  except chameleon.error:
    print('failed to capture')

chameleon.close(h)


