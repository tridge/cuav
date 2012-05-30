#!/usr/bin/env python
'''
test program for block_xmit
'''

import sys, os, time, random, functools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'lib'))
import block_xmit
            

from optparse import OptionParser
parser = OptionParser("block_xmit.py [options]")
parser.add_option("--receive", action='store_true', default=False, help="run as receiver")
parser.add_option("--port", type='int', default=1966, help="udp port to use")
parser.add_option("--dest-ip", default='', help="destination IP to send to")
parser.add_option("--listen-ip", default='', help="IP to listen on")
parser.add_option("--bandwidth", type='int', default=100000, help="bandwidth to use in bytes/sec")
parser.add_option("--chunk-size", type='int', default=1000, help="chunk size to use in bytes")
parser.add_option("--backlog", type='int', default=100, help="number of in-flight chunks")
parser.add_option("--loss", type='float', default=0.0, help="packet loss")
parser.add_option("--debug", action='store_true', default=False, help="verbose debug")
parser.add_option("--count", type='int', default=1, help="number of blocks to send")
(opts, args) = parser.parse_args()

import hashlib

bs = block_xmit.BlockSender(opts.port,
			    dest_ip=opts.dest_ip,
			    listen_ip=opts.listen_ip,
			    bandwidth=opts.bandwidth, 
			    chunk_size=opts.chunk_size,
			    backlog=opts.backlog,
			    debug=opts.debug)

if opts.loss:
	print("set packet loss of %.1f%%" % (100.0*opts.loss))
	bs.set_packet_loss(opts.loss)
	
if opts.receive:
    while True:
        data = bs.recv(0.01)
        if data is not None:
            print("received %u bytes sum %s" % (len(data), hashlib.md5(data).hexdigest()))
        bs.tick()


total_size = 0
count = 0
completion_count = 0
start_count = 0

def completed(i):
	global completion_count
	print("completed transfer %u" % i)
	completion_count += 1

print('starting')
bs.reset_timer()
t0 = time.time()
while completion_count < opts.count:
    if bs.sendq_size() < 10 and start_count < opts.count:
        buf = os.urandom(random.randint(1,200000))
        total_size += len(buf)
        print('sending buffer of length %u with sum %s' % (len(buf), hashlib.md5(buf).hexdigest()))
        bs.send(buf, callback=functools.partial(completed, start_count))
	start_count += 1
    bs.tick()
    time.sleep(0.02)
t1 = time.time()
print("Sent %u bytes in %.1f seconds - %.1f bytes/s" % (total_size, t1-t0, total_size/(t1-t0)))
print('rtt_estimate=%f' % bs.rtt_estimate)
