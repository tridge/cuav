#!/usr/bin/env python
'''
test program for block_xmit
'''

import sys, os, time, random, functools
import hashlib
from argparse import ArgumentParser
from cuav.lib import block_xmit

def completed(i):
    global completion_count
    print("completed transfer %u" % i)
    completion_count += 1
    
if __name__ == '__main__':
    parser = ArgumentParser(description="Test program from block_xmit")
    parser.add_argument("--receive", action='store_true', default=False, help="run as receiver")
    parser.add_argument("--port", type=int, default=1966, help="udp port to use")
    parser.add_argument("--dest-ip", default='', help="destination IP to send to")
    parser.add_argument("--listen-ip", default='', help="IP to listen on")
    parser.add_argument("--bandwidth", type=int, default=100000, help="bandwidth to use in bytes/sec")
    parser.add_argument("--chunk-size", type=int, default=1000, help="chunk size to use in bytes")
    parser.add_argument("--backlog", type=int, default=100, help="number of in-flight chunks")
    parser.add_argument("--loss", type=float, default=0.0, help="packet loss")
    parser.add_argument("--debug", action='store_true', default=False, help="verbose debug")
    parser.add_argument("--mss", type=int, default=0, help="maximum segment size")
    parser.add_argument("--count", type=int, default=1, help="number of blocks to send")
    args = parser.parse_args()

    bs = block_xmit.BlockSender(args.port,
                    dest_ip=args.dest_ip,
                    listen_ip=args.listen_ip,
                    bandwidth=args.bandwidth, 
                    chunk_size=args.chunk_size,
                    backlog=args.backlog,
                    mss=args.mss,
                    debug=args.debug)

    if args.loss:
        print("set packet loss of %.1f%%" % (100.0*args.loss))
        bs.set_packet_loss(args.loss)
        
    if args.receive:
        while True:
            data = bs.recv(0.01)
            if data is not None:
                print("received %u bytes sum %s" % (len(data), hashlib.md5(data).hexdigest()))
            bs.tick()


    total_size = 0
    count = 0
    completion_count = 0
    start_count = 0


    print('starting')
    bs.reset_timer()
    t0 = time.time()
    while completion_count < args.count:
        if bs.sendq_size() < 10 and start_count < args.count:
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
    print("efficiency %.1f  bandwidth used %.1f bytes/s" % (bs.get_efficiency(),
                                bs.get_bandwidth_used()))
