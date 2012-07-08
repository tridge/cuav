#!/usr/bin/python
'''
Outback Challenge Mission Generator
It read in a KML file and generate mission waypoints (ie. search pattern)
Created by Stephen Dade (stephen_dade@hotmail.com)
'''

import numpy, os, time, sys, xml.dom.minidom
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'MAVProxy', 'modules', 'lib'))
import mp_slipmap, mp_util

class MissionGenerator():
    '''Mission Generator Class'''

    def __init__(self, inFile='obc.kml'):
        self.inFile = inFile
        self.dom = xml.dom.minidom.parse(inFile)

    def Process(self):
        '''Processes the imported xml file for points'''
        
        self.searchArea = []
        self.missionBounds = []
        #get a list of all points in the kml file:
        airf = self.dom.getElementsByTagName('Placemark')
        for point in airf:
            if self.getElement(point.getElementsByTagName('name')[0]) == "Airfield Home":
                self.airfieldHome = (float(self.getElement(point.getElementsByTagName('latitude')[0])), float(self.getElement(point.getElementsByTagName('longitude')[0])))
                print "Airfield Home = " + str(self.airfieldHome)
            if "SA-" in self.getElement(point.getElementsByTagName('name')[0]):
                self.searchArea.append((float(self.getElement(point.getElementsByTagName('latitude')[0])), float(self.getElement(point.getElementsByTagName('longitude')[0]))))
            if "MB-" in self.getElement(point.getElementsByTagName('name')[0]):
                self.missionBounds.append((float(self.getElement(point.getElementsByTagName('latitude')[0])), float(self.getElement(point.getElementsByTagName('longitude')[0]))))

        print "Search Area = " + str(self.searchArea)
        print "Mission Boundary = " + str(self.missionBounds)

        #get the bounding box of the search area
        self.boundingBoxLat = [(self.searchArea[0])[0], (self.searchArea[0])[0]]
        self.boundingBoxLong = [(self.searchArea[0])[1], (self.searchArea[0])[1]]
        for point in self.searchArea:
            if point[0] < self.boundingBoxLat[0]:
                self.boundingBoxLat[0] = point[0]
            if point[0] > self.boundingBoxLat[1]:
                self.boundingBoxLat[1] = point[0]       
            if point[1] < self.boundingBoxLong[0]:
                self.boundingBoxLong[0] = point[1]
            if point[1] > self.boundingBoxLong[1]:
                self.boundingBoxLong[1] = point[1]   
        #print "Bounding box is: " + str(self.boundingBoxLat) + " ... " + str(self.boundingBoxLong)

        #for point in self.searchArea:
        #    print self.getDistAndBearing(self.airfieldHome, point)


    def CreateSearchPattern(self, width=50.0, overlap=10.0):
        '''Generate the waypoints for the search pattern, using alternating strips
        width is the width (m) of each strip
        overlap is the % overlap between strips'''
        self.SearchPattern = []

        #find the nearest point to Airfield Home - use this as a starting point
        nearestdist = self.getDistAndBearing(self.airfieldHome, self.searchArea[0])[0]
        nearest = self.searchArea[0]
        for point in self.searchArea:
            newdist = self.getDistAndBearing(self.airfieldHome, point)[0]
            if newdist < nearestdist:
                nearest = point
                nearestdist = newdist

        print "Start = " + str(nearest) + ", dist = " + str(nearestdist)

        #the search pattern will then run parallel between the two furthest points in the list
        searchLine = (0, 0)
        for point in self.searchArea: 
            newdist = self.getDistAndBearing(point, self.searchArea[self.searchArea.index(point)-1])
            if newdist[0] > searchLine[0]:
                searchLine = newdist

        self.searchBearing = searchLine[1]
        print "Search bearing is " + str(self.searchBearing) + "/" + str((self.searchBearing + 180) % 360)

        #need to find the 90 degree bearing to searchBearing that is inside the search area. This
        #will be the bearing we increment the search rows by
        if not self.polygon_outside(self.addDistBearing(nearest, 10, (self.searchBearing + 45) % 360), self.searchArea):
            self.crossBearing = (self.searchBearing + 90) % 360
        else:
            self.crossBearing = (self.searchBearing - 90) % 360
        print "Cross bearing is: " + str(self.crossBearing)

        #function check
        newposn = self.addDistBearing(nearest, nearestdist, 358)
        print "Going back to start: " + str(newposn)

        #the distance between runs is this:
        self.deltaRowDist = width - width*(overlap/100)
        if self.deltaRowDist <= 0:
            print "Error, overlap % is too high"
            return

        #we are starting at the "nearest" and mowing the lawn parallel to "self.searchBearing"
        #first waypoint is width*(overlap/100) on an opposite bearing (so behind the search area)
        nextWaypoint =  self.addDistBearing(nearest, width*(overlap/100), self.crossBearing)
        print "First = " + str(nextWaypoint)
        #self.SearchPattern.append(firstWaypoint)

        #mow the lawn, every 2nd row:
        while True:
            pts = self.projectBearing(self.searchBearing, nextWaypoint, self.searchArea)
            #check if we're outside the search area
            if pts == 0:
                break
            (nextW, nextnextW) = (pts[0], pts[1])
            if self.getDistAndBearing(nextWaypoint, nextW)[0] < self.getDistAndBearing(nextWaypoint, nextnextW)[0]:
                self.SearchPattern.append(nextW)
                self.SearchPattern.append(nextnextW)
                #now turn 90degrees from bearing and width distance
                nextWaypoint = self.addDistBearing(nextnextW, width*2, self.crossBearing)
            else:
                self.SearchPattern.append(nextnextW)
                self.SearchPattern.append(nextW)
                #now turn 90degrees from bearing and width distance
                nextWaypoint = self.addDistBearing(nextW, width*2, self.crossBearing)

            print "Next = " + str(nextWaypoint)
        
        #go back and do the rows we missed
        self.crossBearing = (self.crossBearing + 180) % 360
        nextWaypoint = self.addDistBearing(self.SearchPattern[-1], width, self.crossBearing)
        while True:
            pts = self.projectBearing(self.searchBearing, nextWaypoint, self.searchArea)
            #check if we're outside the search area
            if pts == 0:
                break
            (nextW, nextnextW) = (pts[0], pts[1])
            if self.getDistAndBearing(nextWaypoint, nextW)[0] < self.getDistAndBearing(nextWaypoint, nextnextW)[0]:
                self.SearchPattern.append(nextW)
                self.SearchPattern.append(nextnextW)
                #now turn 90degrees from bearing and width distance
                nextWaypoint = self.addDistBearing(nextnextW, width*2, self.crossBearing)
            else:
                self.SearchPattern.append(nextnextW)
                self.SearchPattern.append(nextW)
                #now turn 90degrees from bearing and width distance
                nextWaypoint = self.addDistBearing(nextW, width*2, self.crossBearing)

            print "Next = " + str(nextWaypoint)

    def isOutsideSearchAreaBoundingBox(self, lat, longy):
        '''Checks if the long/lat pair is inside the search area
        bounding box. Returns true if it is inside'''
        if lat < self.boundingBoxLat[0]:
            return 1
        if lat > self.boundingBoxLat[1]:
            return 1
        if longy < self.boundingBoxLong[0]:
            return 1
        if longy > self.boundingBoxLong[1]:
            return 1
        return 0

    def ExportSearchPattern(self, filename='search.kml'):
        '''Exports the current search pattern to Google Earth kml file'''
        if len(self.SearchPattern) == 0:
            return

        #and actually write the file
        with open(filename, 'w') as f:
            f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            f.write("<kml xmlns=\"http://www.opengis.net/kml/2.2\" xmlns:gx=\"http://www.google.com/kml/ext/2.2\" xmlns:kml=\"http://www.opengis.net/kml/2.2\">\n")
            f.write("<Document>\n")
            f.write("    <name>" + filename + "</name>\n")

            #loop around the waypoints
            for point in self.SearchPattern:
                f.write("    <Placemark>\n")
                f.write("        <name>SP-" + str(self.SearchPattern.index(point)) + "</name>\n")
                f.write("        <Point>\n")
                f.write("            <coordinates>" + str(point[1]) + "," + str(point[0]) + ",0</coordinates>\n")
                f.write("        </Point>\n")
                f.write("    </Placemark>\n")

            #make a polygon for easier viewing
            f.write("    <Placemark>\n")
            f.write("        <name>Search Pattern</name>\n")
            f.write("        <LineString>\n")
            f.write("            <tessellate>0</tessellate>\n")
            f.write("            <coordinates>\n")
            for point in self.SearchPattern:
                f.write("                " + str(point[1]) + "," + str(point[0]) + ",0\n")
            f.write("            </coordinates>\n")
            f.write("        </LineString>\n")
            f.write("    </Placemark>\n")
            
            f.write("</Document>\n")
            f.write("</kml>")


    def getElement(self, element):
        return self.getText(element.childNodes)


    def getText(self, nodelist):
        rc = ""
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc = rc + node.nodeValue
        return rc


    def getDistAndBearing(self, startPosn, endPosn):
        '''Returns the distance (m) and bearing (deg from north) from a lat/long tuple to
        an ending lat/long tuple'''

        #use the sperical law of cosines formula for distance
        dist = numpy.arccos(numpy.sin(numpy.radians(startPosn[0]))*numpy.sin(numpy.radians(endPosn[0]))+ \
               numpy.cos(numpy.radians(startPosn[0]))*numpy.cos(numpy.radians(endPosn[0]))* \
               numpy.cos(numpy.radians(endPosn[1])-numpy.radians(startPosn[1])))*6372.79*1000

        #this formula taken from www.movable-type.co.uk/scripts/latlong.html
        bearing = numpy.arctan2(numpy.sin(numpy.radians(endPosn[1])-numpy.radians(startPosn[1]))*numpy.cos(numpy.radians(endPosn[0])), \
                                numpy.cos(numpy.radians(startPosn[0]))*numpy.sin(numpy.radians(endPosn[0])) - \
                                numpy.sin(numpy.radians(startPosn[0]))*numpy.cos(numpy.radians(endPosn[0]))* \
                                numpy.cos(numpy.radians(endPosn[1])-numpy.radians(startPosn[1])))

        #format bearing to a degrees compass heading
        bearing = (numpy.degrees(bearing) + 360) % 360

        return (dist, bearing)

    def addDistBearing(self, startPosn, dist, bearing):
        '''Travels dist and bearing from the startPosn (lat/long tuple) and
        returns the ending posn (lat/long tuple)'''

        #this formula taken from www.movable-type.co.uk/scripts/latlong.html
        lat = numpy.arcsin(numpy.sin(numpy.radians(startPosn[0]))*numpy.cos(dist/(6372.79*1000))+numpy.cos(numpy.radians(startPosn[0]))*numpy.sin(dist/(6372.79*1000))*numpy.cos(numpy.radians(bearing)))
        deltaLong = numpy.arctan2(numpy.sin(numpy.radians(bearing))*numpy.sin(dist/(6372.79*1000))*numpy.cos(numpy.radians(startPosn[0])), \
                    numpy.cos(dist/(6372.79*1000))-numpy.sin(numpy.radians(startPosn[0]))*numpy.sin(lat))
        longitude = numpy.radians(startPosn[1]) + deltaLong

        return (numpy.degrees(lat), numpy.degrees(longitude))

    def polygon_outside(self, P, V):
	    '''return true if point is outside polygon
	    P is a (x,y) tuple
	    V is a list of (x,y) tuples
        
	    The point in polygon algorithm is based on:
	    http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
	    '''
	    n = len(V)
	    outside = True
	    j = n-1
	    for i in range(n):
		    if (((V[i][1]>P[1]) != (V[j][1]>P[1])) and
		        (P[0] < (V[j][0]-V[i][0]) * (P[1]-V[i][1]) / (V[j][1]-V[i][1]) + V[i][0])):
			    outside = not outside
		    j = i
	    return outside

    def projectBearing(self, bearing, startPos, searchArea):
        '''Projects bearing from startPos until it reaches the edge(s) of
        searchArea (list of lat/long tuples. Returns the First/Last position(s)'''

        coPoints = []

        #for each line in the search are border, get any overlaps with startPos on bearing(+-180)
        for point in searchArea:
            (dist, theta2) = self.getDistAndBearing(point, searchArea[searchArea.index(point)-1])
            posn = self.Intersection(startPos, bearing, point, theta2)
            if posn != 0 and self.getDistAndBearing(posn, point)[0] < dist:
                coPoints.append(posn)
            posn = self.Intersection(startPos, (bearing + 180) % 360, point, theta2)
            if posn != 0 and self.getDistAndBearing(posn, point)[0] < dist:
                coPoints.append(posn)

        #if there's more than two points in coPoints, return the furthest away points
        if len(coPoints) < 2:
            return 0
        elif len(coPoints) == 2:
            return coPoints
        else:
            dist = 0
            for point in coPoints:
                for pt in coPoints:
                    if self.getDistAndBearing(pt, point)[0] > dist:
                        dist = self.getDistAndBearing(pt, point)[0]
                        newcoPoints = [point, pt]
            return newcoPoints


    def Intersection(self, pos1, theta1, pos2, theta2):
        '''This will find the intersection between two positions and bearings
           it will return the intersection lat/long, or 0 if no intersection'''
        brng1 = theta1
        brng2 = theta2
        lat1 = numpy.radians(pos1[0])
        lon1 = numpy.radians(pos1[1])
        lat2 = numpy.radians(pos2[0])
        lon2 = numpy.radians(pos2[1])
        brng13 = numpy.radians(brng1)
        brng23 = numpy.radians(brng2)
        dLat = lat2-lat1
        dLon = lon2-lon1
  
        dist12 = 2*numpy.arcsin( numpy.sqrt( numpy.sin(dLat/2)*numpy.sin(dLat/2) + numpy.cos(lat1)*numpy.cos(lat2)*numpy.sin(dLon/2)*numpy.sin(dLon/2) ) )
        if dist12 == 0:
            return 0
  
        #initial/final bearings between points
        brngA = numpy.arccos( ( numpy.sin(lat2) - numpy.sin(lat1)*numpy.cos(dist12) ) / ( numpy.sin(dist12)*numpy.cos(lat1) ) )
        #if (isNaN(brngA)) brngA = 0;  // protect against rounding
        brngB = numpy.arccos( ( numpy.sin(lat1) - numpy.sin(lat2)*numpy.cos(dist12) ) / ( numpy.sin(dist12)*numpy.cos(lat2) ) )
  
        if numpy.sin(lon2-lon1) > 0:
            brng12 = brngA
            brng21 = 2*numpy.pi - brngB
        else:
           brng12 = 2*numpy.pi - brngA
           brng21 = brngB
  
        alpha1 = (brng13 - brng12 + numpy.pi) % (2*numpy.pi) - numpy.pi  # angle 2-1-3
        alpha2 = (brng21 - brng23 + numpy.pi) % (2*numpy.pi) - numpy.pi  # angle 1-2-3
  
        if numpy.sin(alpha1)==0 and numpy.sin(alpha2)==0:
            return 0  # infinite intersections
        if numpy.sin(alpha1)*numpy.sin(alpha2) < 0:
            return 0      # ambiguous intersection
   
        alpha3 = numpy.arccos( -numpy.cos(alpha1)*numpy.cos(alpha2) + numpy.sin(alpha1)*numpy.sin(alpha2)*numpy.cos(dist12) )
        dist13 = numpy.arctan2( numpy.sin(dist12)*numpy.sin(alpha1)*numpy.sin(alpha2), numpy.cos(alpha2)+numpy.cos(alpha1)*numpy.cos(alpha3) )
        lat3 = numpy.arcsin( numpy.sin(lat1)*numpy.cos(dist13) + numpy.cos(lat1)*numpy.sin(dist13)*numpy.cos(brng13) )
        dLon13 = numpy.arctan2( numpy.sin(brng13)*numpy.sin(dist13)*numpy.cos(lat1), numpy.cos(dist13)-numpy.sin(lat1)*numpy.sin(lat3) )
        lon3 = lon1+dLon13
        lon3 = (lon3+3*numpy.pi) % (2*numpy.pi) - numpy.pi  # normalise to -180..+180
  
        return (numpy.degrees(lat3), numpy.degrees(lon3));

    def getMapPolygon(self):
        '''Returns a mp_Slipmap compatible polygon'''
        return self.SearchPattern

    def getPolygonLength(self)
        '''Returns the path length

if __name__ == "__main__":

    from optparse import OptionParser
    parser = OptionParser("mp_missiongenerator.py [options]")
    parser.add_option("--file", type='string', default='OBC Waypoints.kml', help="input file")

    (opts, args) = parser.parse_args()

    inFile = opts.file

    gen = MissionGenerator(inFile)
    gen.Process()
    gen.CreateSearchPattern()
    gen.ExportSearchPattern()

    #start a map
    sm = mp_slipmap.MPSlipMap(lat=gen.getMapPolygon()[0][0], lon=gen.getMapPolygon()[0][1], elevation=True)
    sm.add_object(mp_slipmap.SlipPolygon('Search Pattern', gen.getMapPolygon(), layer=1, linewidth=2, colour=(0,255,0)))

    #test function
    #posn = gen.Intersection((51.885, 0.235), 108.63, (49.008, 2.549), 32.72)
    #print "Intersection test = " + str(posn)



