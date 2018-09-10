#!/usr/bin/env python

# gegpsd-0.3.py retrieved from https://warwick.ac.uk/fac/sci/csc/people/computingstaff/jaroslaw_zachwieja/gegpsd/gegpsd-0.3.py

import gps
import optparse

parser = optparse.OptionParser("gegpsd.py [options]")
parser.add_option("--output", help="Output file", default='/tmp/nmea.kml')
parser.add_option("--server", help="host:port", default="localhost:2947")
parser.add_option("--pin-image", help="pin image", default=None)

(opts, args) = parser.parse_args()

file = opts.output

(host, port) = opts.server.split(":")

session = gps.gps(host=host, port=port)

session.stream(flags=gps.WATCH_JSON)

for report in session:
   if report['class'] == 'TPV':
       latitude  = report['lat']
       longitude = report['lon']
       altitude = report['alt']
       speed_in = report['speed']
       heading = report['track']

       speed = int(speed_in * 1.852)
       range = ( ( speed / 100  ) * 350 ) + 650
       tilt = ( ( speed / 120 ) * 43 ) + 30

       if speed < 10:
           range = 200
           tilt = 30
           heading = 0

       pin_image = ""
       if opts.pin_image is not None:
           pin_image = """	<Style id="mystyle">
		<IconStyle>
			<scale>1.3</scale>
			<Icon>
				<href>%s</href>
			</Icon>
		</IconStyle>
	</Style>
""" % opts.pin_image

       output = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.0">
	<Placemark>
		<name>%s km/h</name>
		<description>^</description>
		<LookAt>
			<longitude>%s</longitude>
			<latitude>%s</latitude>
			<range>%s</range>
			<tilt>%s</tilt>
			<heading>%s</heading>
		</LookAt>
		<Point>
			<coordinates>%s,%s,%s</coordinates>
		</Point>
	%s
	</Placemark>
</kml>""" % (speed,longitude,latitude,range,tilt,heading,longitude,latitude,altitude, pin_image)

       f=open(file, 'w')
       f.write(output)
       f.close()

del session

