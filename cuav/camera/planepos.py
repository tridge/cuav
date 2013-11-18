#!/usr/bin/env python

'''
work out plane position for files in joe.txt
'''

import sys, struct, time, os, math

from pymavlink import mavutil

from optparse import OptionParser
parser = OptionParser("planepos.py [options]")
(opts, args) = parser.parse_args()

if len(args) < 2:
    print("Usage: planepos.py [options] <LOGFILE> <joe.txt>")
    sys.exit(1)

logfile = args[0]
joetxt = args[1]

gps = []
hud = []
attitude = []

ground_height = -1

def process_msg(m, t):
    '''process one mavlink msg'''
    global ground_height
    mtype = m.get_type()
    if mtype == 'GPS_RAW':
        gps.append((t, m))
        if m.fix_type == 2 and ground_height == -1:
            ground_height = m.alt
    elif mtype == 'VFR_HUD' and ground_height != -1:
        hud.append((t, m))
    elif mtype == 'ATTITUDE':
        attitude.append((t, m))

def find_msg(array, t):
    imin = 0
    imax = len(array)-1
    while imin < imax:
        i = (imin+imax)/2
        if i == imin:
            i = imin+1
        (a_t, a_m) = array[i]
        if t > a_t:
            imin = i
        else:
            imax = i-1
    return imin

def interpolate(array, t, i, attr):
    (t1, m1) = array[i]
    (t2, m2) = array[i+1]
    v1 = getattr(m1, attr)
    v2 = getattr(m2, attr)
    return v1 + (((t-t1)/(t2-t1))*(v2-v1))


f = open(logfile, mode='r')

# create a mavlink instance, which will do IO on file object 'f'
mav = mavlink.MAVLink(None)
mav.robust_parsing = True

while True:
    tbuf = f.read(8)
    if len(tbuf) != 8:
        break
    (tusec,) = struct.unpack('>Q', tbuf)
    t = tusec / 1.0e6

    # read the packet
    while True:
        c = f.read(1)
        if c == "":
            break
        m = mav.parse_char(c)
        if m:
            process_msg(m, t)
            break

f.close()

f = open(joetxt, mode='r')
for line in f:
    line = line.strip()
    a = line.split(" ")
    if len(a) != 3:
        continue
    bname = os.path.basename(a[0])

    t = time.strptime(bname[-21:-7], "%Y%m%d%H%M%S")
    hsec = int(bname[-6:-4])
    t = time.mktime(t) + (hsec * 0.01)

    i = find_msg(gps, t)
    lat = interpolate(gps, t, i, 'lat')
    lon = interpolate(gps, t, i, 'lon')
    hdg = interpolate(gps, t, i, 'hdg')

    i = find_msg(hud, t)
    alt = interpolate(hud, t, i, 'alt') - ground_height

    i = find_msg(attitude, t)
    pitch = interpolate(attitude, t, i, 'pitch')
    roll  = interpolate(attitude, t, i, 'roll')
    yaw   = interpolate(attitude, t, i, 'yaw')

    print("%s %s %s %f %f %f %f %f %f" % (
        a[0], a[1], a[2], lat, lon, alt, math.degrees(yaw), math.degrees(pitch), math.degrees(roll)))
