#!/usr/bin/env python

'''
modify a mission to do terrain following at a specified altitude above ground level
'''

import sys, time, os, copy

from MAVProxy.modules.mavproxy_map import mp_elevation
from cuav.lib import cuav_util


from optparse import OptionParser
parser = OptionParser("wpagl.py [options]")
parser.add_option("--output", default='mission.txt', help="output file")
parser.add_option("--agl", type='float', default=123, help="AGL")
parser.add_option("--maxdelta", type='float', default=50, help="maximum height delta")
parser.add_option("--speed", type='float', default=11, help="flight speed")
parser.add_option("--step", type='float', default=50, help="wp step distance")
parser.add_option("--rtlalt", type='float', default=100, help="wp step distance")
parser.add_option("--lookahead", type='float', default=150, help="ground lookahead distance")
parser.add_option("--maxclimb", type='float', default=3, help="maximum climb rate")
parser.add_option("--home", default=None, help="new home")

(opts, args) = parser.parse_args()

from pymavlink import mavutil, mavwp


if len(args) < 1:
    print("Usage: mavmission.py [options] <missionfile>")
    sys.exit(1)

parms = {}

EleModel = mp_elevation.ElevationModel()

def get_ground_alt(lat, lon):
    '''get highest ground altitide around a point'''
    global EleModel
    ground = EleModel.GetElevation(lat, lon)
    surrounds = []
    for bearing in range(0, 360, 45):
        surrounds.append((opts.lookahead, bearing))
    for (dist, bearing) in surrounds:
        (lat2, lon2) = cuav_util.gps_newpos(lat, lon, bearing, dist)
        el = EleModel.GetElevation(lat2, lon2)
        if el > ground:
            ground = el
    return ground
    

def fix_alt(filename, agl):
    '''fix AGL on mission'''
    wp = mavwp.MAVWPLoader()
    wp.load(filename)
    home = wp.wp(0)
    if opts.home:
        a = opts.home.split(",")
        home.x = float(a[0])
        home.y = float(a[1])
    home_agl = get_ground_alt(home.x, home.y)
    print("Home AGL %.1f" % home_agl)
    
    for i in range(1, wp.count()):
        w = wp.wp(i)
        if (w.x == 0 and w.y == 0) or w.command not in [mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                                                        mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM,
                                                        mavutil.mavlink.MAV_CMD_NAV_LOITER_TURNS,
                                                        mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME,
                                                        mavutil.mavlink.MAV_CMD_NAV_LAND,
                                                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF]:
            continue
        ground = get_ground_alt(w.x, w.y)
        w.z = (ground - home_agl) + float(agl)
        print("ground Elevation %.1f z=%.1f" % (ground, w.z))

    wp.save(opts.output)
    print("Saved %s" % opts.output)
    return wp

def add_points(wp):
    '''add more points for terrain following'''
    wplist = []
    wplist2 = []
    for i in range(0, wp.count()):
        wplist.append(wp.wp(i))
    wplist[0].z = opts.agl

    # add in RTL
    wplist.append(wplist[0])
    
    wplist2.append(wplist[0])

    home = wp.wp(0)
    home_ground = get_ground_alt(home.x, home.y)

    for i in range(1, len(wplist)):
        prev = (wplist2[-1].x, wplist2[-1].y, wplist2[-1].z)
        dist = cuav_util.gps_distance(wplist2[-1].x, wplist2[-1].y, wplist[i].x, wplist[i].y)
        bearing = cuav_util.gps_bearing(wplist2[-1].x, wplist2[-1].y, wplist[i].x, wplist[i].y)
        print("dist=%u bearing=%u" % (dist, bearing))
        while dist > opts.step:
            newpos = cuav_util.gps_newpos(prev[0], prev[1], bearing, opts.step)
            ground1 = get_ground_alt(prev[0], prev[1])
            ground2 = get_ground_alt(newpos[0], newpos[1])
            agl = (home_ground + prev[2]) - ground2
            if abs(agl - opts.agl) > opts.maxdelta:
                newwp = copy.copy(wplist2[-1])
                newwp.x = newpos[0]
                newwp.y = newpos[1]
                newwp.z = (ground2 + opts.agl) - home_ground
                wplist2.append(newwp)
                print("Inserting at %u" % newwp.z)
                prev = (newpos[0], newpos[1], newwp.z)
            else:
                prev = (newpos[0], newpos[1], wplist2[-1].z)
            dist -= opts.step
        wplist2.append(wplist[i])
    wplist2[-1].z = opts.rtlalt
    wp2 = mavwp.MAVWPLoader()
    for w in wplist2:
        wp2.add(w)
    wp2.save("newwp.txt")
    return wp2

def fix_climb(wp):
    '''fix waypoints for max climb rate'''
    while True:
        adjusted = False
        for i in range(1, wp.count()):
            w0 = wp.wp(i-1)
            w = wp.wp(i)
            w.frame = 3
            wp_dist = cuav_util.gps_distance(w0.x, w0.y, w.x, w.y)
            wp_time = wp_dist / opts.speed
            climb = (w.z - w0.z) / wp_time
            if climb > opts.maxclimb+0.1 and i > 1:
                z = w.z - (wp_time * opts.maxclimb)
                adjusted = True
                print("fix climb %.1f i=%u by %.1fm" % (climb, i, z-w0.z))
                w0.z = z
        if not adjusted:
            return wp
    return wp

def report_points(wp):
    '''show points agl'''

    home = wp.wp(0)
    home_ground = get_ground_alt(home.x, home.y)

    total_distance = 0
    max_climb_rate = 0

    for i in range(1, wp.count()):
        w0 = wp.wp(i-1)
        w = wp.wp(i)
        ground2 = get_ground_alt(w.x, w.y)
        agl = (home_ground + w.z) - ground2
        print("wp[%u] agl=%u" % (i, agl))
        wp_dist = cuav_util.gps_distance(w0.x, w0.y, w.x, w.y)
        wp_time = wp_dist / opts.speed
        climb = (w.z - w0.z) / wp_time
        if climb > max_climb_rate and i > 1:
            max_climb_rate = climb
            print("Climb %.1f at wp %u" % (climb, i))
        total_distance += wp_dist
    t = total_distance / opts.speed
    print("Total_distance: %.2f km  flight time %u:%u max_climb=%.1f" % (
        total_distance*0.001,
        int(t/60),
        t % 60,
        max_climb_rate))
                                                      
    
total = 0.0
for filename in args:
    wp = fix_alt(filename, opts.agl)
    wp = add_points(wp)
    wp = fix_climb(wp)
    wp.save('newwp.txt')
    report_points(wp)
    

