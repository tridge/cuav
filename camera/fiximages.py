#!/usr/bin/env python

import util, os

leader = ""
for i in range(0, 512):
    leader += chr(0)

def fix_image(f):
    '''fix one image'''
    img = util.PGM(f)
    if img.comment and img.comment.find("FIXED1024") != -1:
        return
    print("Fixing image %s" % f)
    os.rename(f, f+'.old')
    f2 = open(f, mode='w')
    f2.write('''P5
1280 960
%s (FIXED1024)
65535
''' % img.comment.strip())
    global leader
    f2.write(leader)
    img.rawdata.byteswap(True)
    f2.write(img.rawdata)
    f2.close()
    if not opts.keep:
        os.unlink(f+'.old')

from optparse import OptionParser
parser = OptionParser("fiximages.py [options] <filename...>")
parser.add_option('-k', '--keep', dest="keep", action='store_true', default=False,
                  help="keep old images")
(opts, args) = parser.parse_args()

for f in args:
    fix_image(f)
