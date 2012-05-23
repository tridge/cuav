#!/usr/bin/env python
'''visualise camera projection onto ground
'''

import cuav_util, numpy, sys, time
from matplotlib import pyplot

from optparse import OptionParser
parser = OptionParser("test_projection.py [options]")
parser.add_option("--roll", type='float', default=0.0, help="roll in degrees")
parser.add_option("--pitch", type='float', default=0.0, help="pitch in degrees")
parser.add_option("--yaw", type='float', default=0.0, help="yaw in degrees")
parser.add_option("--altitude", type='float', default=100.0, help="altitude in meters")
parser.add_option("--lens", type='float', default=4.0, help="lens in mm")
parser.add_option("--xres", type='int', default=640, help="X resolution")
parser.add_option("--yres", type='int', default=480, help="Y resolution")
parser.add_option("--step", type='int', default=16, help="display grid resolution")
(opts, args) = parser.parse_args()


f = pyplot.figure(1)
f.clf()

minx = -200
maxx = 200
miny = -200
maxy = 200

# show position below plane
pyplot.plot(0, 0, 'ro')

total_time = 0
count = 0

for x in range(0, opts.xres, opts.step):
    for y in range(0, opts.yres, opts.step):
        t0 = time.time()
        (ofs_x, ofs_y) = cuav_util.pixel_position(x, y,
                                                  opts.altitude,
                                                  opts.pitch, opts.roll, opts.yaw,
                                                  opts.lens,
                                                  xresolution=opts.xres, 
                                                  yresolution=opts.yres)
        t1 = time.time()
        total_time += t1 - t0
        count += 1
        minx = min(minx, ofs_x)
        miny = min(miny, ofs_y)
        maxx = max(maxx, ofs_x)
        maxy = max(maxy, ofs_y)
        color = 'bo'
        if (x,y) == (0,0):
            # show corner pixel in yellow
            color = 'yo'
        pyplot.plot(ofs_x, ofs_y, color)

print("Speed: %.1f projections/second" % (count/total_time))
        
pyplot.axis([minx-50,maxx+50, miny-50, maxy+50])
f.show()

cuav_util.cv_wait_quit()

