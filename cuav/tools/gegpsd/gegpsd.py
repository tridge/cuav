#!env python

# gegpsd-0.3.py retrieved from https://warwick.ac.uk/fac/sci/csc/people/computingstaff/jaroslaw_zachwieja/gegpsd/gegpsd-0.3.py

import gps

file = '/tmp/nmea.kml'

session = gps.gps(host="localhost", port="2947")

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
	</Placemark>
</kml>""" % (speed,longitude,latitude,range,tilt,heading,longitude,latitude,altitude)

       f=open(file, 'w')
       f.write(output)
       f.close()

del session

