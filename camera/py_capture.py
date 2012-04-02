#!/usr/bin/python

from numpy import array,zeros
import chameleon

colour = 1
depth = 8
h = chameleon.open('/dev/chameleon', colour, depth)
im = zeros((960,1280),dtype='uint8')
(shutter, ftime) = chameleon.capture(h, im)
chameleon.close()
