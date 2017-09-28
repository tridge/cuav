#!/usr/bin/env python
'''visualise camera projection onto ground
'''

import numpy, sys, time, os
from matplotlib import pyplot

from cuav.lib import cuav_util
from cuav.camera import cam_params

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
parser.add_option("--border", type='int', default=200, help="border size")
parser.add_option("--cam-params", default=None, help="camera parameters file")
(opts, args) = parser.parse_args()


f = pyplot.figure(1)
f.clf()

minx = 0
maxx = 0
miny = 0
maxy = 0

# show position below plane
pyplot.plot(0, 0, 'yo')

total_time = 0
count = 0

includes_sky = False

print("Roll=%.1f Pitch=%.1f Yaw=%.1f Altitude=%.1f" % (opts.roll, opts.pitch, opts.yaw, opts.altitude))

if opts.cam_params is not None:
    C_params = cam_params.CameraParams.fromfile(opts.cam_params)
else:
    C_params = cam_params.CameraParams(sensorwidth=5.0,
                                       lens=opts.lens,
                                       xresolution=opts.xres,
                                       yresolution=opts.yres)

def plot_point(x, y):
    '''add one point'''
    global total_time, count, minx, maxx, miny, maxy, C_params
    t0 = time.time()

    ofs = cuav_util.pixel_position_matt(x, y,
                                        opts.altitude,
                                        opts.pitch, opts.roll, opts.yaw,
                                        C_params)
    if ofs is None:
        includes_sky = True
        return
    (ofs_x, ofs_y) = ofs
    t1 = time.time()
    total_time += (t1 - t0)
    count += 1
    minx = min(minx, ofs_x)
    miny = min(miny, ofs_y)
    maxx = max(maxx, ofs_x)
    maxy = max(maxy, ofs_y)
    color = 'bo'
    if x < opts.xres/4 and y < opts.yres/4:
        # show corner in red
        color = 'ro'
    pyplot.plot(ofs_x, ofs_y, color)

for x in range(C_params.xresolution):
    plot_point(x, 0)
    plot_point(x, C_params.yresolution/2)
    plot_point(x, C_params.yresolution-1)
for y in range(C_params.yresolution):
    plot_point(0, y)
    plot_point(C_params.xresolution/2, y)
    plot_point(C_params.xresolution-1, y)
        
if includes_sky:
    print("Projection includes sky")

print("Speed: %.1f projections/second" % (count/total_time))

print("Range: ", minx, miny, maxy, maxy)

minx = min(minx, -opts.border)
miny = min(miny, -opts.border)
maxx = max(maxx, opts.border)
maxy = max(maxy, opts.border)
        
pyplot.axis([minx-50,maxx+50, miny-50, maxy+50])
f.show()

cuav_util.cv_wait_quit()

