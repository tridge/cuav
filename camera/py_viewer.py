#!/usr/bin/python

import chameleon, cv, time, sys, os, numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'image'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'lib'))
import scanner, cuav_util

from optparse import OptionParser
parser = OptionParser("py_viewer.py [options]")
parser.add_option("--depth", type='int', default=8, help="capture depth")
parser.add_option("--output", default='tmp', help="output directory")
parser.add_option("--half", action='store_true', default=False, help="show half res")
parser.add_option("--brightness", type='float', default=1.0, help="display brightness")
parser.add_option("--gamma", type='int', default=1024, help="set gamma (if 16 bit images)")
(opts, args) = parser.parse_args()


colour = 0
try:
  h = chameleon.open(1, opts.depth, 100)
  colour = 1
except chameleon.error:
  h = chameleon.open(0, opts.depth, 100)
  colour = 0

print("Found camera: colour=%u GUID=%x" % (colour, chameleon.guid(h)))
if opts.depth == 8:
  dtype = 'uint8'
else:
  dtype = 'uint16'
im = numpy.zeros((960,1280),dtype=dtype)

cv.NamedWindow('Viewer')

tstart = time.time()

chameleon.trigger(h, True)
chameleon.set_gamma(h, opts.gamma)

cuav_util.mkdir_p(opts.output)

i=0
lost = 0
while True:
  try:
    frame_time, frame_counter, shutter = chameleon.capture(h, 300, im)
  except chameleon.error, msg:
    lost += 1
    continue
  filename = '%s/i%u.pgm' % (opts.output, i)
  chameleon.save_pgm(filename, im)
  img_colour = numpy.zeros((960,1280,3),dtype='uint8')
  scanner.debayer_full(im, img_colour)
  img_colour = cv.GetImage(cv.fromarray(img_colour))
  if opts.half:
    img_640 = cv.CreateImage((640,480), 8, 3)
    cv.Resize(img_colour, img_640)
    img_colour = img_640

  cv.ConvertScale(img_colour, img_colour, scale=opts.brightness)
  cv.ShowImage('Viewer', img_colour)
  key = cv.WaitKey(1)
  i += 1

  if i % 10 == 0:
    tdiff = time.time() - tstart
    print("captured %u  lost %u  %.1f fps shutter=%f" % (
      i, lost,
      10/tdiff, shutter));
    tstart = time.time()

  key = cv.WaitKey(1)
  if key == -1:
    continue
  if key == ord('q'):
    break

chameleon.close(h)
cv.DestroyWindow('Viewer')

