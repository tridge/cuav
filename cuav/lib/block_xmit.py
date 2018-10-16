#!/usr/bin/env python
'''
a module for reliable block data sending over UDP

NOTE: This module should only be used on private networks - it takes
no account of network congestion.

The protocol is designed to work well with large amounts of packet loss, while
using a fixed maximum bandwidth. The actual send bandwidth scales closely
as the target bandwidth times the packet loss.

The protocol sends arbitrary "blocks" of data. It was designed for
sending images and telemetry data efficiently over a lossy wireless
link. The default transport is UDP, but the user can specify their own
message oriented transport if needed. 

Andrew Tridgell
May 2012
released under the GNU GPL v3 or later
'''

import socket, select, os, random, time, random, struct, binascii

# packet types - first byte of a packet
PKT_ACK = 0
PKT_COMPLETE = 1
PKT_CHUNK = 2

# size of packet type plus crc32
PACKET_HEADER_SIZE = 5

class BlockSenderException(Exception):
    '''block sender error class'''
    def __init__(self, msg):
        Exception.__init__(self, msg)


class BlockSenderSet:
    '''hold a set of chunk IDs for an identifier.
    This object is sent as a PKT_ACK to
    acknowledge receipt of data'''
    def __init__(self, id, num_chunks, mss):
        self.id = id
        self.num_chunks = num_chunks
        self.chunks = set()
        self.timestamp = 0
        self.format = '<QHd'
        self.header_size = struct.calcsize(self.format)
        self.first_missing = 0
        self.mss = mss
        self.last_sent = 0
                #print("Created %s" % str(self))

    def __str__(self):
        return 'BlockSenderSet<%u/%u>' % (len(self.chunks), self.num_chunks)

    def update_first_missing(self):
        '''update the first_missing field'''
        while self.first_missing < self.num_chunks:
            if not self.first_missing in self.chunks:
                break
            self.first_missing += 1

    def add(self, chunk_id, ack_to):
        '''add an extent to the list. This is called when we receive a chunk of data'''
        self.chunks.add(chunk_id)
        self.first_missing = ack_to

    def update(self, new):
        '''add in new chunks. This is called when we receive an ack packet'''
        self.chunks.update(new.chunks)
        self.update_first_missing()

    def present(self, chunk_id):
        '''see if a chunk_id is present in the chunks'''
        return chunk_id in self.chunks

    def complete(self):
        '''return True if the chunks cover the whole set of data'''
        return len(self.chunks) == self.num_chunks

    def started(self):
        '''return True if we have at least one chunk'''
        return len(self.chunks) > 0

    def pack(self):
        '''return a linearized representation'''
        chunks = list(self.chunks)
        chunks.sort()
        extents = []
        for i in range(len(chunks)):
            if chunks[i] < self.first_missing:
                continue
            if len(extents) == 0:
                extents.append((chunks[i], 1))
                continue
            (first,count) = extents[-1]
            if chunks[i] == first+count:
                extents[-1] = (first, count+1)
            else:
                extents.append((chunks[i], 1))
        buf = bytes(struct.pack(self.format, self.id, self.num_chunks, self.timestamp))
        if self.mss:
            max_extents = (self.mss - (len(buf) + PACKET_HEADER_SIZE)) / 2
            if max_extents > len(extents):
                # not all of the extents will fit. Use last_sent to choose which ones
                # to send
                while len(extents) > max_extents:
                    (first,count) = extents[0]
                    if first > self.last_sent:
                        break
                    extents.pop(0)
        sent_all = True
        for (first,count) in extents:
            buf += bytes(struct.pack('<HH', first, count))
            self.last_sent = first
            if self.mss and (len(buf)+PACKET_HEADER_SIZE) + 4 > self.mss:
                sent_all = False
                break
        if sent_all:
            self.last_sent = 0
        return buf

    def unpack(self, buf):
        '''unpack a linearized representation into the object'''
        if len(buf) < self.header_size:
            raise BlockSenderException('buffer too short')
        (self.id, self.num_chunks, self.timestamp) = struct.unpack_from(self.format, buf)
        ofs = self.header_size
        if (len(buf) - ofs) % 4 != 0:
            raise BlockSenderException('invalid extents length')
        n = (len(buf) - ofs) // 4
        for i in range(n):
            (first, count) = struct.unpack_from('<HH', buf, ofs)
            ofs += 4
            for j in range(first, first+count):
                self.chunks.add(j)


class BlockSenderComplete:
    '''a packet to say that a block is complete.
    This is a bit more efficient than sending a complete extents list'''
    def __init__(self, blockid, timestamp, dest):
        self.blockid = blockid
        self.timestamp = timestamp
        self.dest = dest
        #print("Created %s" % str(self))

    def __str__(self):
        return 'BlockSenderComplete<%u>' % self.blockid

    def pack(self):
        '''return a linearized representation'''        
        return bytes(struct.pack('<Qd', self.blockid, self.timestamp))

    def unpack(self, buf):
        '''unpack a linearized representation into the object'''
        (self.blockid, self.timestamp) = struct.unpack('<Qd', buf)


class BlockSenderChunk:
    '''an incoming chunk packet. This is the main data format'''
    def __init__(self, blockid, size, chunk_id, data, chunk_size, ack_to, timestamp):
        self.blockid = blockid
        self.size = size
        self.chunk_id = chunk_id
        self.chunk_size = chunk_size
        self.data = data
        self.ack_to = ack_to
        self.timestamp = timestamp
        self.format = '<QLHHHd'
        self.header_size = struct.calcsize(self.format)
        if data is not None:
            self.packed_size = len(data) + self.header_size
        else:
            self.packed_size = 0
        #print("Created %s" % str(self))

    def __str__(self):
        return 'BlockSenderChunk<%u,%u,%u,%u>' % (self.blockid, self.chunk_id, self.size, self.chunk_size)

    def pack(self):
        '''return a linearized representation'''        
        buf = bytes(struct.pack(self.format, self.blockid, self.size, self.chunk_id,
                        self.chunk_size, self.ack_to, self.timestamp))
        buf += bytes(self.data)
        return buf

    def unpack(self, buf):
        '''unpack a linearized representation into the object'''
        (self.blockid, self.size,
         self.chunk_id, self.chunk_size, self.ack_to, self.timestamp) = struct.unpack_from(self.format, buf, offset=0)
        self.data = bytes(buf[self.header_size:])


class BlockSenderBlock:
    '''the state of an incoming or outgoing block'''
    def __init__(self, blockid, size, chunk_size, dest, mss, data=None, callback=None, priority=0):
        self.blockid = blockid
        self.size = size
        self.chunk_size = chunk_size
        self.num_chunks = (self.size + (chunk_size-1)) // chunk_size
        self.acks = BlockSenderSet(blockid, self.num_chunks, mss)
        if data is not None:
            self.data = bytearray(data)
        else:
            self.data = bytearray(size)
        self.timestamp = 0
        self.callback = callback
        self.dest = dest
        self.next_chunk = 0
        self.priority = priority
        self.sends = 0
        self.chunk_send_times = {}
        #print("Created %s" % str(self))

    def __str__(self):
        return 'BlockSenderBlock<%u,%u,%u,%u>' % (self.blockid,self.size,self.chunk_size,self.num_chunks)

    def chunk(self, chunk_id):
        '''return data for a chunk'''
        start = chunk_id*self.chunk_size
        return self.data[start:start+self.chunk_size]        

    def complete(self):
        '''return true if all chunks have been sent/received'''
        return self.acks.complete()


class BlockSender:
    '''a reliable datagram block sender

    port:          UDP port to listen on, use zero for a system allocated port. This
        port can be queried via get_port()
    dest_ip:       default IP to send to
    dest_port:     default port for send, defaults to port
    listen_ip:     IP to listen on (default is wildcard)
    bandwidth:     bandwidth to use in bytes/second (default 100000 bytes/s)
    completed_len: how many completed blocks to remember (default 100)
    chunk_size:    size of data chunks to send in bytes (default 1000)
    backlog:       maximum number of packets to send per tick (default 100)
    rtt:           initial round trip time estimate (0.01 seconds)
    sock:          a optional socket object to use, needs sendto() and recvfrom()
                plus a fileno() method if recv() with non-zero timeout is used
    mss:           maximum segment size for any packet. This limits all
               packet types (default is zero, meaning no limit)
    ordered:       set to True to force blocks to be delivered in the sending order (default False)
    debug:         enable debugging (default False)
    '''
    def __init__(self, port=0, dest_ip=None, dest_port=None, listen_ip='', bandwidth=100000,
             completed_len=1000, chunk_size=1000, backlog=100, rtt=0.01,
             sock=None, mss=0, ordered=False,
             debug=False):
        self.bandwidth = bandwidth
        self.port = port
        if dest_port is None:
            dest_port = port
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((listen_ip, port))
            self.sock.setblocking(False)
            if port == 0:
                (host, self.port) = self.sock.getsockname()
        else:
            self.sock = sock
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.outgoing = []
        self.incoming = []
        self.next_blockid = os.getpid() << 20
        self.last_send_time = time.time()
        self.last_recv_time = time.time()
        self.acks_needed = set()
        self.packet_loss = 0
        self.completed_len = completed_len
        self.completed = []
        if chunk_size > 65535:
            raise BlockSenderException('chunk size must be less than 65536')
        self.chunk_size = chunk_size
        self.enable_debug = debug
        self.backlog = backlog
        self.rtt_estimate = rtt
        self.rtt_offset = 0
        self.rtt_max = 5
        self.rtt_multiplier = 3.0
        self.mss = mss
        self.ordered = ordered
        self.bonus_bytes = 0
        self.efficiency = 1.0
        self.bandwidth_used = 0.0
        self.send_count = 0
        self.recv_count = 0
        self.last_receive_time = 0

        # work out the overheads of the packet types
        self.chunk_overhead = BlockSenderChunk(0,0,0,'',0,0,0).header_size
        self.ack_overhead = BlockSenderSet(0,0,0).header_size
        if self.mss and (self.mss < self.chunk_overhead + 1 or
                 self.mss < self.ack_overhead + 4):
            raise BlockSenderException('mss is too small')

    def get_port(self):
        '''return the port we are receiving on'''
        return self.port

    def set_dest_port(self, port):
        '''set the port we send to by default'''
        self.dest_port = port

    def set_packet_loss(self, loss):
        '''set a percentage packet loss
        This can be used to simulate lossy networks
        '''
        self.packet_loss = loss

    def set_bandwidth(self, bandwidth):
        '''set the bandwidth on an open sender'''
        self.bandwidth = bandwidth

    def get_efficiency(self):
        '''return the average efficiency of the link. An efficiency of 1.0 means
        each chunk is sent just once. An efficiency of 0.2 means each chunk is
        sent an average of 5 times'''
        return self.efficiency

    def get_rtt_estimate(self):
        '''return an estimate of the round trip time'''
        return self.rtt_estimate

    def get_bandwidth_used(self):
        '''return a moving average of the actual bandwidth used'''
        return self.bandwidth_used

        def is_alive(self, timeout):
            '''return True if link has received a packet in last timeout seconds'''
            return time.time() - self.last_receive_time < timeout
                

    def send(self, data, dest=None, chunk_size=None, callback=None, priority=0):
        '''send a data block

        dest:       optional (host,port) tuple
        chunk_size: network send size for this block (defaults to self.chunk_size)
        callback:   optional callback function on completion of send (default None)
        priority:   optional priority for sending this packet. Higher priority packets
                    are sent first (default 0)

                returns blockid for sent block, which may be passed to cancel()
        '''
        if not chunk_size:
            chunk_size = self.chunk_size
        if self.mss and chunk_size > self.chunk_overhead + self.mss:
            chunk_size = self.mss - (self.chunk_overhead + PACKET_HEADER_SIZE)

        num_chunks = (len(data) + (chunk_size-1)) // chunk_size
        if num_chunks > 65535:
            raise BlockSenderException('chunk_size of %u is too small for data length %u' % (chunk_size, len(data)))
        blockid = self.next_blockid
        self.next_blockid += 1
        if dest is None:
            if self.dest_ip is None:
                raise BlockSenderException('no destination specified in send')
            dest = (self.dest_ip, self.dest_port)

        newblk = BlockSenderBlock(blockid, len(data), chunk_size, dest, self.mss,
                                  data=data, callback=callback, priority=priority)

        # if this block has a non-zero priority, insert after the last one with a
        # higher or equal priority
        if priority > 0:
            for i in range(len(self.outgoing), 0, -1):
                if self.outgoing[i-1].priority >= priority:
                                        #print("Inserted blk %u len=%u %s" % (i, len(self.outgoing), newblk))
                    self.outgoing.insert(i, newblk)
                    return newblk.blockid
                    #print("Inserted blk %u len=%u %s" % (0, len(self.outgoing), newblk))
            self.outgoing.insert(0, newblk)
            return newblk.blockid
        # otherwise append to the outgoing list
        #print("Appended blk len=%u %s" % (len(self.outgoing), newblk))
        self.outgoing.append(newblk)
        return newblk.blockid

    def cancel(self, blockid):
        '''cancel send of a block

        blockid:    id of block returned from send()
        '''
        for i in range(len(self.outgoing)):
            if self.outgoing[i].blockid == blockid:
                self.outgoing.pop(i)
                self._debug('Cancelled block %u' % blockid)
                return

    def _crc(self, buffer):
        '''produce a 32 bit unsigned crc for a buffer'''
        return binascii.crc32(bytes(buffer)) & 0xFFFFFFFF

    def _debug(self, s):
        '''internal debug function'''
        if self.enable_debug:
            print('%.3f %s' % (time.time(), s))

    def _send_object(self, obj, type, dest):
        '''low level object send'''
        if self.packet_loss != 0:
            if random.uniform(0, 1) < self.packet_loss*0.01:
                                #print("lose packet")
                return
        try:
            buf = obj.pack()
            crc = self._crc(buf)
            buf = bytes(struct.pack('<BL', type, crc)) + buf
            self.sock.sendto(buf, dest)
            self.send_count += 1
            #print("send_count=%u %s" % (self.send_count, obj))
        except socket.error:
            pass

    def _send_acks(self):
        '''send extents objects to acknowledge data'''
        tnow = time.time()
        deltat = tnow - self.last_recv_time
        self.last_recv_time = tnow
        if self.acks_needed and self.enable_debug:
            print("sending %u acks deltat=%.2f" % (len(self.acks_needed), deltat))
        acks_needed = self.acks_needed.copy()
        for obj in acks_needed:
            try:
                if isinstance(obj, BlockSenderBlock):
                    obj.acks.timestamp = obj.timestamp
                    if obj.complete():
                        ack = BlockSenderComplete(obj.blockid, obj.timestamp, obj.dest)
                        self._send_object(ack, PKT_COMPLETE, obj.dest)
                    else:
                        pkt = obj.acks
                        self._send_object(obj.acks, PKT_ACK, obj.dest)
                else:
                    (blockid, dest) = obj
                    ack = BlockSenderComplete(blockid, time.time(), dest)
                    self._send_object(ack, PKT_COMPLETE, dest)
                self.acks_needed.remove(obj)
            except Exception as e:
                self._debug('_send_acks: ' + str(e))
                return

    def _add_chunk(self, blk, chunk):
        '''add an incoming chunk to a block'''
        blk.acks.add(chunk.chunk_id, chunk.ack_to)
        start = chunk.chunk_id*chunk.chunk_size
        length = len(chunk.data)
        blk.data[start:start+length] = chunk.data
        self.acks_needed.add(blk)

    def _complete_send(self, blk):
        '''complete send of a block'''
        if blk.callback:
                        #print("Callback %s" % blk.callback)
            blk.callback()
        if blk.sends == 0:
            efficiency = blk.num_chunks / 1.0
        else:
            efficiency = blk.num_chunks / float(blk.sends)
        #print("_complete_send: efficiency=%.2f sends=%u recvs=%u" % (efficiency, self.send_count, self.recv_count))
        self.efficiency = 0.95 * self.efficiency + 0.05 * efficiency

    def _update_rtt(self, obj, tnow):
        '''update rtt estimate from a received packet'''
        if self.rtt_offset + tnow < obj.timestamp:
            # the two clocks are not in sync. Use the negative round trip time to adjust
            self.rtt_offset = obj.timestamp - tnow
            self._debug("rtt_offset=%.3f" % self.rtt_offset)
        self.rtt_estimate = min(self.rtt_max, 0.95 * self.rtt_estimate + 0.05 * (self.rtt_offset + tnow - obj.timestamp))

    def _check_incoming(self):
        '''check for incoming data or acks. Return True if a packet was received'''
        try:
            (buf, fromaddr) = self.sock.recvfrom(65536)
        except socket.error:
            return False
        if len(buf) == 0:
            return False
        self.recv_count += 1
        if self.dest_ip is None:
            if self.enable_debug:
                self._debug('connection from %s' % str(fromaddr))
            # setup defaults for send based on first connection
            (self.dest_ip,self.dest_port) = fromaddr
        try:
            if len(buf) < PACKET_HEADER_SIZE:
                self._debug('bad packet %s' % msg)
                return True
            (magic,crc) = struct.unpack_from('<BL', buf)
            remaining = buf[PACKET_HEADER_SIZE:]
            if crc != self._crc(remaining):
                self._debug('bad crc')
                return True                
            if magic == PKT_ACK:
                obj = BlockSenderSet(0,0,0)
                obj.unpack(remaining)
            elif magic == PKT_COMPLETE:
                obj = BlockSenderComplete(0, None, None)
                obj.unpack(remaining)
            elif magic == PKT_CHUNK:
                obj = BlockSenderChunk(0, 0, 0, "", 0, 0, 0)
                obj.unpack(remaining)
            else:
                self._debug('bad magic %u' % magic)
                return True
        except Exception as e:
            self._debug('_check_incoming: bad packet %s' % str(e))
            return True
        tnow = time.time()
        self.last_receive_time = tnow
        #print(obj)

        if isinstance(obj, BlockSenderSet):
            # we've received a set of acks for some data
            # find the corresponding outgoing block
            self._update_rtt(obj, tnow)
            for i in range(len(self.outgoing)):
                out = self.outgoing[i]
                if out.blockid == obj.id:
                    if self.enable_debug:
                        self._debug("ack %s %f" % (str(out.acks), self.rtt_offset + tnow - obj.timestamp))
                    out.acks.update(obj)
                    if out.acks.complete():
                        if self.enable_debug:
                            self._debug("send complete %u %s" % (out.blockid, obj))
                        blk = self.outgoing.pop(i)
                        self._complete_send(blk)
                    return True
            # an ack for something already complete
            return True

        if isinstance(obj, BlockSenderComplete):
            # a full block has been received
            if self.enable_debug:
                self._debug("full ack for blockid %u" % obj.blockid)
                self._update_rtt(obj, tnow)
            for i in range(len(self.outgoing)):
                out = self.outgoing[i]
                if out.blockid == obj.blockid:
                    blk = self.outgoing.pop(i)
                    if self.enable_debug:
                        self._debug("send complete %u outlen=%u %s %s" % (
                                                        out.blockid, len(self.outgoing), obj, blk))
                    self._complete_send(blk)
                    return True
            # an ack for something already complete
            return True

        if isinstance(obj, BlockSenderChunk):
            # we've received a chunk of data
            if obj.blockid in self.completed:
                # we've already completed this blockid
                if self.enable_debug:
                    self._debug("got completed chunk %u of %u" % (obj.chunk_id, obj.blockid))
                self.acks_needed.add((obj.blockid, fromaddr))
                return True
            for i in range(len(self.incoming)):
                blk = self.incoming[i]
                if blk.blockid == obj.blockid:
                    # we have an existing incoming object
                    if self.enable_debug:
                        if obj.chunk_id in blk.acks.chunks:
                            self._debug("got dup chunk %u of %u" % (obj.chunk_id, obj.blockid))
                        else:
                            self._debug("got chunk %u of %u" % (obj.chunk_id, obj.blockid))
                    blk.timestamp = obj.timestamp
                    self._add_chunk(blk, obj)
                    return True
            # its a new block
            if self.enable_debug:
                self._debug("new block chunk %u of %u (size=%u chunk_size=%u)" % (
                                        obj.chunk_id, obj.blockid, obj.size, obj.chunk_size))
            self.incoming.append(BlockSenderBlock(obj.blockid, obj.size, obj.chunk_size, fromaddr, self.mss))
            blk = self.incoming[-1]
            blk.timestamp = obj.timestamp
            self._add_chunk(blk, obj)
            return True
        self._debug("unexpected incoming packet type")
        return True


    def available(self, ordered=None):
        '''return the first incoming block if completed or None

        This does no network operations
        '''
        if ordered is None:
            ordered = self.ordered
        imax = len(self.incoming)
        if ordered:
            imax = min(1, imax)
        for i in range(imax):
            if self.incoming[i].complete():
                blk = self.incoming.pop(i)
                #print("available sends=%u recvs=%u" % (self.send_count, self.recv_count))
                self.completed.append(blk.blockid)
                while len(self.completed) > self.completed_len:
                    self.completed.pop(0)        
                return blk.data
        return None

    def report(self, detailed=False):
        '''report chunk status'''
        total_acked = 0
        total_chunks = 0
        for i in range(len(self.outgoing)):
            blk = self.outgoing[i]
            total_acked += len(blk.acks.chunks)
            total_chunks += blk.acks.num_chunks
            if detailed:
                print("block %u  acked %u/%u" % (blk.blockid, len(blk.acks.chunks), blk.acks.num_chunks))
                complete = "0"
                if len(self.incoming) > 0:
                        complete = "%u/%u" % (len(self.incoming[0].acks.chunks), self.incoming[0].acks.num_chunks)
                print("total_acked=%u total_chunks=%u eff=%.2f rtt=%.1f bw=%.2f qsize=%u in=%u/%s" % (
                        total_acked, total_chunks, self.get_efficiency(), self.get_rtt_estimate(),
                        self.get_bandwidth_used(),
                        self.sendq_size(), len(self.incoming), complete))

    def sendq_size(self):
        '''return number of uncompleted blocks in the send queue'''
        return len(self.outgoing)


    def recv(self, timeout=0, ordered=None):
        '''receive next chunk from network. Return data or None

        timeout:  time to wait for a packet (0 means to return immediately)
        ordered:  return blocks in same order as sent (default False)
        '''
        if ordered is None:
            ordered = self.ordered
        data = self.available(ordered=ordered)
        if data is not None:
            return data
        if len(self.incoming) > 0 and self.incoming[0].complete():
            return self.incoming.pop(0).data            
        if timeout != 0:
            rin = [self.sock.fileno()]
            try:
                (rin, win, xin) = select.select(rin, [], [], timeout)
            except select.error:
                return None
        self._check_incoming()
        return self.available(ordered=ordered)


    def reset_timer(self):
        '''reset the timer used for bandwidth control'''
        self.last_send_time = time.time()


    def _send_outgoing(self, max_queue=None):
        '''send any outgoing data that is due to be sent'''
        if len(self.outgoing) == 0:
            return

        tnow = time.time()
        deltat = tnow - self.last_send_time
        bytes_to_send = int(self.bandwidth * deltat + self.bonus_bytes)

        # don't try and send till we have a reasonable amount we can send. On higher bandwidth links
        # this makes sending more efficient by using bigger chunks
        if bytes_to_send <= self.bandwidth/10:
            return
        bytes_sent = 0
        chunks_sent = 0

        count = len(self.outgoing)
        if max_queue is not None:
            count = min(max_queue, count)

        for i in range(count):
            blk = self.outgoing[i]

            # in order to preserve ordering, we have to make sure the other end
            # has acked at least one chunk from the previous block before moving
            # to the next block
            if self.ordered and i > 0 and not self.outgoing[i-1].acks.started():
                break

            # start where we left off
            chunks = list(range(blk.next_chunk, blk.num_chunks))
            chunks.extend(range(blk.next_chunk))

            for c in chunks:
                if blk.acks.present(c):
                    # we've received an ack for this chunk already
                    continue
                if bytes_sent + blk.chunk_size > bytes_to_send:
                    # this would take us over our bandwidth limit
                    break
                if c in blk.chunk_send_times:
                    if tnow - blk.chunk_send_times[c] < self.rtt_multiplier*self.rtt_estimate:
                        # wait for a possible ack
                        continue

                chunk = BlockSenderChunk(blk.blockid, blk.size, c, blk.chunk(c),
                             blk.chunk_size, blk.acks.first_missing, tnow)

                if bytes_sent + chunk.packed_size > bytes_to_send:
                    # this would take us over our bandwidth limit
                    break

                if self.enable_debug:
                    self._debug('send chunk len=%u dt=%.3f bts=%u bsent=%u bonus=%u' % (
                        chunk.packed_size, deltat, bytes_to_send, bytes_sent, self.bonus_bytes))
                try:
                    self._send_object(chunk, PKT_CHUNK, blk.dest)
                except Exception as e:
                    self._debug('_send_outgoing: ' + str(e))
                    break
                                #print("sent chunk size=%u of %u sends=%u blockid=%u" % (
                                #        chunk.chunk_size, blk.size, blk.sends, blk.blockid))
                bytes_sent += chunk.packed_size
                blk.next_chunk = (c + 1) % blk.num_chunks
                blk.timestamp = tnow
                blk.sends += 1
                chunks_sent += 1
                blk.chunk_send_times[c] = tnow
                if chunks_sent == self.backlog:
                    # don't send more than self.backlog per tick
                    break

        # adjust bonus, but don't allow it to get too far ahead
        self.bonus_bytes = bytes_to_send - bytes_sent
        self.bonus_bytes = min(self.bonus_bytes, self.bandwidth//2)
                
        self.last_send_time = tnow
        if bytes_sent != 0:
            self.bandwidth_used = 0.99 * self.bandwidth_used + 0.01 * (bytes_sent/deltat)


    def tick(self, packet_count=None, send_acks=True, send_outgoing=True, max_queue=None):
        '''periodic timer to trigger data sends

        This should be called regularly to process incoming packets, send acks and send any
        pending data

        packet_count:  maximum number of incoming packets to process (default self.backlog)
        send_acks:     send acknowledgement packets (default True)
        send_outgoing: send outgoing packets (default True)
        '''
        # check for incoming packets
        if packet_count is None:
            packet_count = self.backlog
        for i in range(packet_count):
            if not self._check_incoming():
                break

        # send any acks that are needed
        if send_acks:
            self._send_acks()

        # send outgoing data
        if send_outgoing:
            self._send_outgoing(max_queue=max_queue)

# a simple test suite
if __name__ == "__main__":
    import sys

    print("block_xmit test")

    debug = False
    bandwidth = 100000
    ordered = False
    num_blocks = 100
    packet_loss = 15
    average_block_size = 50000

    # setup a send/recv pair
    b1 = BlockSender(dest_ip='127.0.0.1', debug=debug, bandwidth=bandwidth, ordered=ordered)
    b2 = BlockSender(dest_ip='127.0.0.1', debug=debug, bandwidth=bandwidth, ordered=ordered)

    # setup for some packet loss
    if packet_loss:
        b1.set_packet_loss(packet_loss)
        b2.set_packet_loss(packet_loss)

    # tell b1 the port b2 got allocated
    print("using ports %u and %u" % (b1.get_port(), b2.get_port()))
    b1.set_dest_port(b2.get_port())
    b2.set_dest_port(b1.get_port())

    # generate some data blocks to work with
    blocks = [bytes(os.urandom(random.randint(1,average_block_size))) for i in range(num_blocks)]

    total_size = sum([len(blk) for blk in blocks])

    print("sending %u bytes as %u blocks bandwidth=%u packet_loss=%.1f%%" %
          (total_size, len(blocks), bandwidth, packet_loss))

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

    t1 = time.time()

    if not ordered:
        blocks.sort()
        received1.sort()
        received2.sort()

    if blocks != received1:
        print("ERROR: received1 blocks not equal to sent")
        sys.exit(1)
    if blocks != received2:
        print("ERROR: received2 blocks not equal to sent")
        sys.exit(1)

    print("%u blocks received OK %.1f bytes/second" % (num_blocks, total_size/(t1-t0)))
    print("efficiency %.1f  bandwidth used %.1f bytes/s" % (b1.get_efficiency(),
                                b1.get_bandwidth_used()))
