#!/usr/bin/env python

import math
import argparse


def aov(sensorwidth, lens):
    '''return angle of view in degrees'''
    return math.degrees(2.0*math.atan((sensorwidth/1000.0)/(2.0*lens/1000.0)))

def groundwidth(height, sensorwidth, lens):
    '''return frame width on ground'''
    return 2.0*height*math.tan(math.radians(aov(sensorwidth, lens)/2))

def pixelwidth(xresolution, height, sensorwidth, lens):
    '''return pixel width on ground in meters'''
    return groundwidth(height, sensorwidth, lens)/xresolution;

def pixelarea(xresolution, height, sensorwidth, lens):
    '''return pixel width on ground in meters'''
    return math.pow(pixelwidth(xresolution, height, sensorwidth, lens), 2)

def lamparea(lampdiameter):
    '''return lamp area in m^2'''
    return math.pi*math.pow((lampdiameter/100.0)/2.0, 2.0)

def lamppower(lampefficiency, arglamppower):
    '''return lamp power over its area'''
    return (lampefficiency/100.0) * arglamppower;

def lamppixelpower(lampefficiency, arglamppower, xresolution, lampdiameter, height, sensorwidth, lens):
    '''return lamp power in W over a pixel area'''
    if pixelarea(xresolution, height, sensorwidth, lens) > lamparea(lampdiameter):
        return lamppower(lampefficiency, arglamppower)
    return lamppower(lampefficiency, arglamppower) * (pixelarea(xresolution, height, sensorwidth, lens)/lamparea(lampdiameter))

def sunonlamp(illumination, lampdiameter):
    '''return sunlight over lamp area in W'''
    return illumination*lamparea(lampdiameter)

def sunreflected(albedo, illumination, xresolution, height, sensorwidth, lens):
    '''reflected sunlight over pixel area in W'''
    return albedo * illumination * pixelarea(xresolution, height, sensorwidth, lens)

def apparentbrightness(lampefficiency, arglamppower, xresolution, lampdiameter, filterfactor, albedo, illumination, height, sensorwidth, lens):
    '''apparent brightness of lamp over surroundings'''
    return (lamppixelpower(lampefficiency, arglamppower, xresolution, lampdiameter, height, sensorwidth, lens) + (filterfactor*sunreflected(albedo, illumination, xresolution, height, sensorwidth, lens))) / (filterfactor*sunreflected(albedo, illumination, xresolution, height, sensorwidth, lens))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Lens calculations")
    parser.add_argument("--lens",dest="lens", type=float, default='2.8',
                      help="lens size in mm")
    parser.add_argument("--illumination",dest="illumination", type=float, default='1500',
                      help="sunlight brightness in W/m^2")
    parser.add_argument("--lampdiameter",dest="lampdiameter", type=float, default='6.5',
                      help="lamp diameter in cm")
    parser.add_argument("--lampefficiency",dest="lampefficiency", type=float, default='10',
                      help="lamp efficieny (percentage)")
    parser.add_argument("--lamppower",dest="lamppower", type=float, default='50',
                      help="lamp power in W")
    parser.add_argument("--albedo",dest="albedo", type=float, default='0.2',
                      help="albedo of ground")
    parser.add_argument("--height", dest='height', type=float, default='122',
                      help='height in meters')
    parser.add_argument("--sensorwidth", dest='sensorwidth', type=float, default='5.0',
                      help='sensor width in mm')
    parser.add_argument("--filterfactor", dest='filterfactor', type=float, default='1.0',
                      help='filter pass ratio')
    parser.add_argument("--xresolution", dest='xresolution', type=int, default='1280')
    
    args = parser.parse_args()

    print("Inputs:")
    print("Lens: %.1f mm" % args.lens)
    print("Illumination: %.0f W/m^2" % args.illumination)
    print("Lamp diameter: %.1f cm" % args.lampdiameter)
    print("Lamp efficieny: %.0f %%" % args.lampefficiency)
    print("Lamp power: %.1f W" % args.lamppower)
    print("Ground albedo: %.2f" % args.albedo)
    print("Height: %.1f m" % args.height)
    print("Sensor width: %.1f mm" % args.sensorwidth)
    print("X resolution: %d px" % args.xresolution)
    print("Filter pass ratio: %.1f" % args.filterfactor)

    print("\nOutputs:")
    print("ground width: %.1f m" % groundwidth(args.height, args.sensorwidth, args.lens))
    print("angle of view: %.1f degrees" % aov(args.sensorwidth, args.lens))
    print("ground pixel width: %.2f cm" % (100*pixelwidth(args.xresolution, args.height, args.sensorwidth, args.lens)))
    print("ground pixel area: %.2f cm^2" % (10000*pixelarea(args.xresolution, args.height, args.sensorwidth, args.lens)))
    print("lamp area: %f cm^2" % (10000*lamparea(args.lampdiameter)))
    print("lamp output power: %.1f W" % lamppower(args.lampefficiency, args.lamppower))
    print("lamp output power per pixel: %.1f W" % lamppixelpower(args.lampefficiency, args.lamppower, args.xresolution, args.lampdiameter, args.height, args.sensorwidth, args.lens))
    print("Sunlight power on lamp area: %.2f W" % sunonlamp(args.illumination, args.lampdiameter))
    print("Reflected sunlight over pixel area: %.2f W" % sunreflected(args.albedo, args.illumination, args.xresolution, args.height, args.sensorwidth, args.lens))
    print("Apparent brightness of lamp: %.2f" % apparentbrightness(args.lampefficiency, args.lamppower, args.xresolution, args.lampdiameter, args.filterfactor, args.albedo, args.illumination, args.height, args.sensorwidth, args.lens))
