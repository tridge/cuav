#!/usr/bin/python

import chameleon, numpy, os

colour = 1
depth = 8
h = chameleon.open(colour, depth)

im = numpy.zeros((960,1280),dtype='uint8')

try:
  os.mkdir('tmp')
except os.error:
  pass

for i in range(0,10):
  try:
    chameleon.trigger(h)
    shutter = chameleon.capture(h, im)
    filename = 'tmp/i%u.pgm' % i
    chameleon.save_pgm(h, filename, im)
    print("Captured to %s" % filename)
  except chameleon.error:
    print('failed to capture')

chameleon.close(h)


