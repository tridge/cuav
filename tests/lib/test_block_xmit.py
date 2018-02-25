#!/usr/bin/env python
'''
test program for block_xmit
'''

import sys, os, time, random, functools
import pytest
from cuav.lib import block_xmit
            

def test_block_xmit():

    debug = False
    bandwidth = 100000
    ordered = False
    num_blocks = 10
    packet_loss = 0
    average_block_size = 5000

    # setup a send/recv pair
    b1 = block_xmit.BlockSender(dest_ip='127.0.0.1', debug=debug, bandwidth=bandwidth, ordered=ordered)
    b2 = block_xmit.BlockSender(dest_ip='127.0.0.1', debug=debug, bandwidth=bandwidth, ordered=ordered)

    # setup for some packet loss
    if packet_loss:
        b1.set_packet_loss(packet_loss)
        b2.set_packet_loss(packet_loss)

    # tell b1 the port b2 got allocated
    b1.set_dest_port(b2.get_port())
    b2.set_dest_port(b1.get_port())

    # generate some data blocks to work with
    blocks = [bytes(os.urandom(random.randint(1,average_block_size))) for i in range(num_blocks)]

    total_size = sum([len(blk) for blk in blocks])

    t0 = time.time()

    # send them from b1 to b2 and from b2 to b1
    for blk in blocks:
        b1.send(blk)
        b2.send(blk)

    received1 = []
    received2 = []

    # wait till they have all been received and acked
    while (b1.sendq_size() > 0 or
           b2.sendq_size() > 0 or
           len(received1) != num_blocks or
           len(received2) != num_blocks):
        b1.tick()
        b2.tick()
        blk = b2.recv(0.01)
        if blk is not None:
            received2.append(blk)
        blk = b1.recv(0.01)
        if blk is not None:
            received1.append(blk)
        if int(time.time()) - int(t0) > 2:
            #timeout
            assert False

    t1 = time.time()

    if not ordered:
        blocks.sort()
        received1.sort()
        received2.sort()

    assert blocks == received1
    assert blocks == received2
    

    #print("%u blocks received OK %.1f bytes/second" % (num_blocks, total_size/(t1-t0)))
    #print("efficiency %.1f  bandwidth used %.1f bytes/s" % (b1.get_efficiency(),
    #                          b1.get_bandwidth_used()))
