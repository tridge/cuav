#!/usr/bin/env python

'''
produce flight log book
'''

import sys, time, os, glob
from argparse import ArgumentParser
from pymavlink import mavutil
from pymavlink.mavextra import distance_two


class loginfo(object):
    def __init__(self):
        self.flight_time = 0
        self.distance = 0

class LogEntry:
    def __init__(self, logfile, takeoff_time, flight_time, distance):
        self.logfile = logfile
        self.takeoff_time = takeoff_time
        self.flight_time = flight_time
        self.distance = distance

def add_log_entry(logbook, logfile, takeoff_time, flight_time, distance):
    '''add flight time to a date record'''
    logbook.append(LogEntry(logfile, takeoff_time, flight_time, distance))

def show_logbook(logbook):
    '''display logbook'''
    print("Date              FlightTime(minutes) Distance(km)")
    logbook = sorted(logbook, key=lambda e: e.takeoff_time)
    for e in logbook:
        tstring = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.takeoff_time))
        print("%s                 %4.1f        %5.1f   %s" % (tstring, e.flight_time / 60.0, e.distance / 1e3, e.logfile))

def flight_time(logbook, logfile, argcondition, argmindist, argmintime):
    '''work out flight time for a log file'''
    print("Processing log %s" % logfile)
    mlog = mavutil.mavlink_connection(logfile)

    in_air = False
    start_time = None
    total_time = 0.0
    total_dist = 0.0
    t = None
    last_msg = None

    aircraft = logfile.split('/')[0]
    first_takeoff = None
    status = 0

    while True:
        m = mlog.recv_match(type=['GPS','GPS_RAW_INT','VFR_HUD'], condition=argcondition)
        if m is None:
            if in_air:
                total_time += time.mktime(t) - start_time
            if total_time > 0:
                print("Flight time %s %u:%02u" % (logfile, int(total_time)/60, int(total_time)%60))
            if in_air and total_dist >= argmindist*1000 and total_time >= argmintime*60:
                add_log_entry(logbook, logfile, start_time, total_time, total_dist)
            return (total_time, total_dist)
        if not mlog.motors_armed():
            continue
        if m.get_type() == 'GPS_RAW_INT':
            groundspeed = m.vel*0.01
            status = m.fix_type
        elif m.get_type() == 'GPS':
            groundspeed = m.Spd
            status = m.Status
        elif m.get_type() == 'VFR_HUD':
            throttle = m.throttle
        if status < 3:
            continue
        t = time.localtime(m._timestamp)
        if not in_air:
            in_air = True
            start_time = time.mktime(t)

        if m.get_type() in ['GPS','GPS_RAW_INT']:
            if last_msg is not None:
                total_dist += distance_two(last_msg, m)
            last_msg = m
    if in_air and total_dist >= argmindist*1000 and total_time >= argmintime*60:
        add_log_entry(logfile, start_time, total_time, total_dist)
    return (total_time, total_dist)

if __name__ == '__main__':
    parser = ArgumentParser(description="create a logbook of all flight logs")
    parser.add_argument("--condition", default=None, help="condition for packets")
    parser.add_argument("--mindist", type=int, default=1, help="minimum distance (km)")
    parser.add_argument("--mintime", type=int, default=1, help="minimum time (mins)")
    parser.add_argument("logs", metavar="LOG", nargs="+")

    args = parser.parse_args()
    
    # logbook two level dictionary of loginfo
    # logbook[datetuple][aircraft]
    logbook = []
    
    total_time = 0.0
    total_dist = 0.0
    for filename in args.logs:
        for f in glob.glob(filename):
            (ftime, fdist) = flight_time(logbook, f, args.condition, args.mindist, args.mintime)
            total_time += ftime
            total_dist += fdist

    print("Total time in air: %u:%02u" % (int(total_time)/60, int(total_time)%60))
    print("Total distance travelled: %.1f meters" % total_dist)

    show_logbook(logbook)
