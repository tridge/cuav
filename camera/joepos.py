#!/usr/bin/env python

'''
work out joe position for files in joepos.txt
'''

import util, os, math

from optparse import OptionParser
parser = OptionParser("joepos.py [options]")
parser.add_option("--kml",dest="kml", action='store_true', default=False, help="output as kml")
parser.add_option("--maxroll",dest="maxroll", type='float', default=90.0, help="maximum roll")
parser.add_option("--maxpitch",dest="maxpitch", type='float', default=90.0, help="maximum pitch")
parser.add_option("--border",dest="border", type='int', default=0, help="image border limit")
(opts, args) = parser.parse_args()

if len(args) < 1:
    print("Usage: joepos.py [options] <joepos.txt>")
    sys.exit(1)

joepos = args[0]

latsum = 0
lonsum = 0
count = 0

if opts.kml:
    print('''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
<Document>
	<name>Joe</name>
	<Style id="default+icon=http://maps.google.com/mapfiles/kml/pal3/icon60.png">
		<IconStyle>
			<Icon>
				<href>http://maps.google.com/mapfiles/kml/pal3/icon60.png</href>
			</Icon>
		</IconStyle>
	</Style>
	<StyleMap id="default+nicon=http://maps.google.com/mapfiles/kml/pal3/icon60.png+hicon=http://maps.google.com/mapfiles/kml/pal3/icon52.png">
		<Pair>
			<key>normal</key>
			<styleUrl>#default+icon=http://maps.google.com/mapfiles/kml/pal3/icon60.png</styleUrl>
		</Pair>
		<Pair>
			<key>highlight</key>
			<styleUrl>#default+icon=http://maps.google.com/mapfiles/kml/pal3/icon52.png</styleUrl>
		</Pair>
	</StyleMap>
	<Style id="default+icon=http://maps.google.com/mapfiles/kml/pal3/icon52.png">
		<IconStyle>
			<scale>1.1</scale>
			<Icon>
				<href>http://maps.google.com/mapfiles/kml/pal3/icon52.png</href>
			</Icon>
		</IconStyle>
		<LabelStyle>
			<scale>1.1</scale>
		</LabelStyle>
	</Style>
''')


f = open(joepos, mode='r')
for line in f:
    line = line.strip()
    a = line.split(" ")
    filename = a[0]
    xpos = int(a[1])
    ypos = int(a[2])
    lat = float(a[3])
    lon = float(a[4])
    alt = float(a[5])
    hdg = float(a[6])
    pitch = float(a[7])
    roll = float(a[8])

    if math.fabs(pitch) > opts.maxpitch or math.fabs(roll) > opts.maxroll:
        continue

    if (xpos < opts.border or xpos > 1280 - opts.border or
        ypos < opts.border or ypos > 960 - opts.border):
        continue

    (joe_lat, joe_lon) = util.pixel_coordinates(xpos, ypos, lat, lon, alt, pitch, roll, hdg)
    if opts.kml:
        print('''
	<Placemark>
		<name>%s</name>
		<Point>
			<coordinates>%f,%f,0</coordinates>
		</Point>
	</Placemark>
''' % (os.path.basename(filename), joe_lon, joe_lat))
    else:
        print("%f %f %s" % (joe_lat, joe_lon, filename))

    count += 1
    latsum += joe_lat
    lonsum += joe_lon

if opts.kml:
    print('''</Document>
</kml>
''')
else:
    print("Average: %f %f" % (latsum/count, lonsum/count))
