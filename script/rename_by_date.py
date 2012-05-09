#!/usr/bin/env python

import sys, os, time

print 'hello'

for f in sys.argv[1:]:
    print f
    stat = os.stat(f)
    extension = f[-4:]
    if extension != ".pgm":
        print extension
        continue
    base = f[:-4]
    hundredths = int(stat.st_mtime * 100.0) % 100
    newname = "raw%s%02u" % (time.strftime("%Y%m%d%H%M%S", time.localtime(stat.st_mtime)), hundredths)
    newname = os.path.join(os.path.dirname(f), newname)
    if os.path.exists(base + '.log'):
        print(base + '.log', newname + '.log')
        os.rename(base + '.log', newname + '.log')
    os.rename(base + '.pgm', newname + '.pgm')
    print newname



