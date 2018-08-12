#!/usr/bin/env python
'''
generate tracks of dynamic no fly zone objects for OBC-2018

This outputs the packets on UDP 45454. It does not use true asterix packets as
generating those is a bit tricky, instead it uses a pickled text format which
is parsed equivalently by the mavproxy_asterix module. This allows this script
to be used for testing avoidance, while the same mavproxy_asterix module is
used with real asterix packets
'''

import math, time, socket, pickle
import random, argparse

parser = argparse.ArgumentParser(description='DNFZ generator')
parser.add_argument("--num-aircraft", default=10, type=int, help="number of aircraft")
parser.add_argument("--num-bird-prey", default=10, type=int, help="number of birds of prey")
parser.add_argument("--num-bird-migratory", default=10, type=int, help="number of migratory birds")
parser.add_argument("--num-weather", default=10, type=int, help="number of weather systems")
args = parser.parse_args()

radius_of_earth = 6378100.0 # in meters
outport = 45454

home = (-27.298440,151.290775)
ground_height = 1100.0
region_width = 15000.0

# object types
DNFZ_types = {
    'Aircraft' : 1,
    'Weather' : 20000,
    'BirdMigrating' : 30000,
    'BirdOfPrey' : 40000
}

track_count = 0
    
def wrap_valid_longitude(lon):
    ''' wrap a longitude value around to always have a value in the range
        [-180, +180) i.e 0 => 0, 1 => 1, -1 => -1, 181 => -179, -181 => 179
    '''
    return (((lon + 180.0) % 360.0) - 180.0)    

def gps_newpos(lat, lon, bearing, distance):
    '''extrapolate latitude/longitude given a heading and distance
    thanks to http://www.movable-type.co.uk/scripts/latlong.html
    '''
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing)
    dr = distance/radius_of_earth

    lat2 = math.asin(math.sin(lat1)*math.cos(dr) +
                     math.cos(lat1)*math.sin(dr)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(dr)*math.cos(lat1),
                             math.cos(dr)-math.sin(lat1)*math.sin(lat2))
    return (math.degrees(lat2), wrap_valid_longitude(math.degrees(lon2)))

def gps_distance(lat1, lon1, lat2, lon2):
    '''return distance between two points in meters,
    coordinates are in degrees
    thanks to http://www.movable-type.co.uk/scripts/latlong.html'''
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)
    dLat = lat2 - lat1
    dLon = lon2 - lon1

    a = math.sin(0.5*dLat)**2 + math.sin(0.5*dLon)**2 * math.cos(lat1) * math.cos(lat2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0-a))
    return radius_of_earth * c

class DNFZ:
    '''a dynamic no-fly zone object'''
    def __init__(self, DNFZ_type):
        if not DNFZ_type in DNFZ_types:
            raise('Bad DNFZ type %s' % DNFZ_type)
        self.DNFZ_type = DNFZ_type
        self.pkt = {'category': 0, 'I010': {'SAC': {'val': 4, 'desc': 'System Area Code'}, 'SIC': {'val': 0, 'desc': 'System Identification Code'}}, 'I040': {'TrkN': {'val': 0, 'desc': 'Track number'}}, 'ts': 0, 'len': 25, 'I220': {'RoC': {'val': 0.0, 'desc': 'Rate of Climb/Descent'}}, 'crc': 'B52DA163', 'I130': {'Alt': {'max': 150000.0, 'min': -1500.0, 'val': 0.0, 'desc': 'Altitude'}}, 'I070': {'ToT': {'val': 0.0, 'desc': 'Time Of Track Information'}}, 'I105': {'Lat': {'val': 0, 'desc': 'Latitude in WGS.84 in twos complement. Range -90 < latitude < 90 deg.'}, 'Lon': {'val': 0.0, 'desc': 'Longitude in WGS.84 in twos complement. Range -180 < longitude < 180 deg.'}}, 'I080': {'SRC': {'meaning': '3D radar', 'val': 2, 'desc': 'Source of calculated track altitude for I062/130'}, 'FX': {'meaning': 'end of data item', 'val': 0, 'desc': ''}, 'CNF': {'meaning': 'Confirmed track', 'val': 0, 'desc': ''}, 'SPI': {'meaning': 'default value', 'val': 0, 'desc': ''}, 'MRH': {'meaning': 'Geometric altitude more reliable', 'val': 1, 'desc': 'Most Reliable Height'}, 'MON': {'meaning': 'Multisensor track', 'val': 0, 'desc': ''}}}
        self.speed = 0.0 # m/s
        self.heading = 0.0 # degrees
        self.yawrate = 0.0
        # random initial position and heading
        self.randpos()
        self.setheading(random.uniform(0,360))
        global track_count
        track_count += 1
        self.pkt['I040']['TrkN']['val'] = DNFZ_types[self.DNFZ_type] + track_count
        print("track %u" % self.pkt['I040']['TrkN']['val'])

    def distance_from_home(self):
        lat = self.pkt['I105']['Lat']['val']
        lon = self.pkt['I105']['Lon']['val']
        return gps_distance(lat, lon, home[0], home[1])
        
    def randpos(self):
        '''random initial position'''
        self.setpos(home[0], home[1])
        self.move(random.uniform(0, 360), random.uniform(0, region_width))

    def randalt(self):
        '''random initial position'''
        self.setalt(ground_height + random.uniform(100, 1500))
        
    def move(self, bearing, distance):
        '''move position by bearing and distance'''
        lat = self.pkt['I105']['Lat']['val']
        lon = self.pkt['I105']['Lon']['val']
        (lat, lon) = gps_newpos(lat, lon, bearing, distance)
        self.setpos(lat, lon)
        
    def setpos(self, lat, lon):
        self.pkt['I105']['Lat']['val'] = lat
        self.pkt['I105']['Lon']['val'] = lon

    def getalt(self):
        return self.pkt['I130']['Alt']['val']
        
    def setalt(self, alt):
        self.pkt['I130']['Alt']['val'] = alt

    def setclimbrate(self, climbrate):
        self.pkt['I220']['RoC']['val'] = climbrate

    def setyawrate(self, yawrate):
        self.yawrate = yawrate
        
    def setspeed(self, speed):
        self.speed = speed

    def setheading(self, heading):
        self.heading = heading
        while self.heading > 360:
            self.heading -= 360.0
        while self.heading < 0:
            self.heading += 360.0

    def move(self, heading, distance):
        lat = self.pkt['I105']['Lat']['val']
        lon = self.pkt['I105']['Lon']['val']
        (lat, lon) = gps_newpos(lat, lon, heading, distance)
        self.setpos(lat, lon)        

    def changealt(self, delta_alt):
        alt = self.pkt['I130']['Alt']['val']
        alt += delta_alt
        self.setalt(alt)
        
    def update(self, deltat=1.0):
        self.move(self.heading, self.speed * deltat)
        climbrate = self.pkt['I220']['RoC']['val']
        self.changealt(climbrate * deltat)
        self.setheading(self.heading + self.yawrate * deltat)

    def __str__(self):
        return str(self.pkt)

    def pickled(self):
        return pickle.dumps(self.pkt)

class Aircraft(DNFZ):
    '''an aircraft that flies in a circuit'''
    def __init__(self, speed=30.0, circuit_width=1000.0):
        DNFZ.__init__(self, 'Aircraft')
        self.setspeed(speed)
        self.circuit_width = circuit_width
        self.dist_flown = 0
        self.randalt()

    def update(self, deltat=1.0):
        '''fly a square circuit'''
        DNFZ.update(self, deltat)
        self.dist_flown += self.speed * deltat
        if self.dist_flown > self.circuit_width:
            self.setheading(self.heading + 90)
            self.dist_flown = 0

class BirdOfPrey(DNFZ):
    '''an bird that circles slowly climbing, then dives'''
    def __init__(self):
        DNFZ.__init__(self, 'BirdOfPrey')
        self.setspeed(16.0)
        self.radius = random.uniform(100,200)
        self.time_circling = 0
        self.dive_rate = -30
        self.climb_rate = 5
        self.drift_speed = random.uniform(5,10)
        self.drift_heading = self.heading
        circumference = math.pi * self.radius * 2
        self.circle_time = circumference / self.speed
        self.turn_rate = 360.0 / self.circle_time
        if random.uniform(0,1) < 0.5:
            self.turn_rate = -self.turn_rate

    def update(self, deltat=1.0):
        '''fly circles, then dive'''
        DNFZ.update(self, deltat)
        self.time_circling += deltat
        self.setheading(self.heading + self.turn_rate * deltat)
        self.move(self.drift_heading, self.drift_speed)
        if self.time_circling > self.circle_time:
            if self.getalt() > ground_height:
                self.setclimbrate(self.dive_rate)
            else:
                self.setclimbrate(self.climb_rate)
                self.time_circling = 0
        if self.distance_from_home() > region_width:
            self.randpos()
            self.randalt()

class BirdMigrating(DNFZ):
    '''an bird that circles slowly climbing, then dives'''
    def __init__(self):
        DNFZ.__init__(self, 'BirdMigrating')
        self.setspeed(random.uniform(4,16))
        self.setyawrate(random.uniform(-0.2,0.2))

    def update(self, deltat=1.0):
        '''fly in long curves'''
        DNFZ.update(self, deltat)
        if self.distance_from_home() > region_width:
            self.randpos()
            self.randalt()

class Weather(DNFZ):
    '''a weather system'''
    def __init__(self):
        DNFZ.__init__(self, 'Weather')
        self.setspeed(random.uniform(1,4))
        self.lifetime = random.uniform(300,600)

    def update(self, deltat=1.0):
        '''straight lines, with short life'''
        DNFZ.update(self, deltat)
        self.lifetime -= deltat
        if self.lifetime <= 0:
            self.randpos()
            self.randalt()
            self.lifetime = random.uniform(300,600)
            
    
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.connect(('', outport))

aircraft = []

# some fixed wing aircraft
for i in range(args.num_aircraft):
    aircraft.append(Aircraft(random.uniform(10, 100), 2000.0))

# some birds of prey
for i in range(args.num_bird_prey):
    aircraft.append(BirdOfPrey())

# some migrating birds
for i in range(args.num_bird_migratory):
    aircraft.append(BirdMigrating())

# some weather systems
for i in range(args.num_weather):
    aircraft.append(Weather())
    
while True:
    dt = 1.0
    time.sleep(dt)
    for a in aircraft:
        a.update(dt)
        try:
            sock.send(a.pickled())
        except Exception:
            print("send failed")
            pass

