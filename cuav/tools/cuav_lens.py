#!/usr/bin/env python

import math

from optparse import OptionParser
parser = OptionParser("lens.py [options]")

parser.add_option("--lens",dest="lens", type='float', default='2.8',
                  help="lens size in mm")
parser.add_option("--illumination",dest="illumination", type='float', default='1500',
                  help="sunlight brightness in W/m^2")
parser.add_option("--lampdiameter",dest="lampdiameter", type='float', default='6.5',
                  help="lamp diameter in cm")
parser.add_option("--lampefficiency",dest="lampefficiency", type='float', default='10',
                  help="lamp efficieny (percentage)")
parser.add_option("--lamppower",dest="lamppower", type='float', default='50',
                  help="lamp power in W")
parser.add_option("--albedo",dest="albedo", type='float', default='0.2',
                  help="albedo of ground")
parser.add_option("--height", dest='height', type='float', default='122',
                  help='height in meters')
parser.add_option("--sensorwidth", dest='sensorwidth', type='float', default='5.0',
                  help='sensor width in mm')
parser.add_option("--filterfactor", dest='filterfactor', type='float', default='1.0',
                  help='filter pass ratio')
parser.add_option("--xresolution", dest='xresolution', type='int', default='1280')
    
(opts, args) = parser.parse_args()


def aov():
    '''return angle of view in degrees'''
    return math.degrees(2.0*math.atan((opts.sensorwidth/1000.0)/(2.0*opts.lens/1000.0)))

def groundwidth():
    '''return frame width on ground'''
    return 2.0*opts.height*math.tan(math.radians(aov()/2))

def pixelwidth():
    '''return pixel width on ground in meters'''
    return groundwidth()/opts.xresolution;

def pixelarea():
    '''return pixel width on ground in meters'''
    return math.pow(pixelwidth(), 2)

def lamparea():
    '''return lamp area in m^2'''
    return math.pi*math.pow((opts.lampdiameter/100.0)/2.0, 2.0)

def lamppower():
    '''return lamp power over its area'''
    return (opts.lampefficiency/100.0) * opts.lamppower;

def lamppixelpower():
    '''return lamp power in W over a pixel area'''
    if pixelarea() > lamparea():
        return lamppower()
    return lamppower() * (pixelarea()/lamparea())

def sunonlamp():
    '''return sunlight over lamp area in W'''
    return opts.illumination*lamparea()

def sunreflected():
    '''reflected sunlight over pixel area in W'''
    return opts.albedo * opts.illumination * pixelarea()

def apparentbrightness():
    '''apparent brightness of lamp over surroundings'''
    return (lamppixelpower() + (opts.filterfactor*sunreflected())) / (opts.filterfactor*sunreflected())

print("Inputs:")
print("Lens: %.1f mm" % opts.lens)
print("Illumination: %.0f W/m^2" % opts.illumination)
print("Lamp diameter: %.1f cm" % opts.lampdiameter)
print("Lamp efficieny: %.0f %%" % opts.lampefficiency)
print("Lamp power: %.1f W" % opts.lamppower)
print("Ground albedo: %.2f" % opts.albedo)
print("Height: %.1f m" % opts.height)
print("Sensor width: %.1f mm" % opts.sensorwidth)
print("X resolution: %d px" % opts.xresolution)
print("Filter pass ratio: %.1f" % opts.filterfactor)

print("\nOutputs:")
print("ground width: %.1f m" % groundwidth())
print("angle of view: %.1f degrees" % aov())
print("ground pixel width: %.2f cm" % (100*pixelwidth()))
print("ground pixel area: %.2f cm^2" % (10000*pixelarea()))
print("lamp area: %f cm^2" % (10000*lamparea()))
print("lamp output power: %.1f W" % lamppower())
print("lamp output power per pixel: %.1f W" % lamppixelpower())
print("Sunlight power on lamp area: %.2f W" % sunonlamp())
print("Reflected sunlight over pixel area: %.2f W" % sunreflected())
print("Apparent brightness of lamp: %.2f" % apparentbrightness())
