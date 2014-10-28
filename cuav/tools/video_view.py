#!/usr/bin/env python

import sys, cv, cv2, time

from MAVProxy.modules.lib import mp_image

from optparse import OptionParser
parser = OptionParser("video_view.py [options] <filename>")
parser.add_option("--rate", type='float', default=1.0, help="frame rate to process")
parser.add_option("--start", type='float', default=0.0, help="start time in seconds")
(opts, args) = parser.parse_args()

if len(args) < 1:
    print("please supply an video file name")
    sys.exit(1)

view_image = mp_image.MPImage(title='VideoView',
                              width=640,
                              height=480,
                              mouse_events=True,
                              key_events=True,
                              can_zoom=True,
                              can_drag=True)

vidcap = cv2.VideoCapture(args[0])

fps = vidcap.get(cv.CV_CAP_PROP_FPS)
print('Video at %.3f fps processing at %.3f fps' % (fps, opts.rate))
if fps < opts.rate:
    opts.rate = fps

vidcap.set(cv.CV_CAP_PROP_POS_MSEC, opts.start*1000)
t = opts.start
delta_t = 1.0/opts.rate

while True:
    success,image = vidcap.read()
    if not success:
        break
    (height, width, depth) = image.shape
    img = cv.CreateImageHeader((width, height), 8, 3)
    cv.SetData(img, image)
    view_image.set_image(img, bgr=True)
    if not view_image.is_alive():
        break
    for event in view_image.events():
        pass
    t += delta_t
    ret = vidcap.set(cv.CV_CAP_PROP_POS_MSEC, t*1000)
    time.sleep(0.05)
    print("t=%.2f" % t)

view_image.terminate()
