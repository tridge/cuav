#!/usr/bin/env python
'''
a module for reliable block data sending over UDP

This module should only be used on private networks - it takes no account
of network congestion.

Andrew Tridgell
May 2012
released under the GNU GPL v3 or later
'''

import socket, select, os, random, time, random, struct, binascii

# packet types - first byte of a packet
PKT_ACK = 0
PKT_COMPLETE = 1
PKT_CHUNK = 2

class BlockSenderException(Exception):
	'''block sender error class'''
	def __init__(self, msg):
            Exception.__init__(self, msg)


class BlockSenderSet:
    '''hold a set of chunk IDs for an identifier.
    This object is sent as a PKT_ACK to
    acknowledge receipt of data'''
    def __init__(self, id, num_chunks):
        self.id = id
        self.num_chunks = num_chunks
        self.chunks = set()
        self.timestamp = 0
        self.format = '<QHd'
        self.header_size = struct.calcsize(self.format)

    def __str__(self):
        return 'BlockSenderSet<%u/%u>' % (len(self.chunks), self.num_chunks)

    def add(self, chunk_id):
        '''add an extent to the list'''
        self.chunks.add(chunk_id)

    def update(self, new):
        '''add in new chunks'''
        self.chunks.update(new.chunks)

    def present(self, chunk_id):
        '''see if a chunk_id is present in the chunks'''
        return chunk_id in self.chunks

    def complete(self):
        '''return True if the chunks cover the whole set of data'''
        return len(self.chunks) == self.num_chunks

    def pack(self):
        '''return a linearized representation'''
        chunks = list(self.chunks)
        chunks.sort()
        extents = []
        for i in range(len(chunks)):
            if len(extents) == 0:
                extents.append((chunks[i], 1))
                continue
            (first,count) = extents[-1]
            if chunks[i] == first+count:
                extents[-1] = (first, count+1)
            else:
                extents.append((chunks[i], 1))
        buf = struct.pack(self.format, self.id, self.num_chunks, self.timestamp)
        for (first,count) in extents:
            buf += struct.pack('<HH', first, count)
        return buf

    def unpack(self, buf):
        '''unpack a linearized representation into the object'''
        if len(buf) < self.header_size:
            raise BlockSenderException('buffer too short')
        (self.id, self.num_chunks, self.timestamp) = struct.unpack_from(self.format, buf)
        ofs = self.header_size
        if (len(buf) - ofs) % 4 != 0:
            raise BlockSenderException('invalid extents length')
        n = (len(buf) - ofs) / 4
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

    def pack(self):
        '''return a linearized representation'''        
        return struct.pack('<Qd', self.blockid, self.timestamp)
        
    def unpack(self, buf):
        '''unpack a linearized representation into the object'''
        (self.blockid, self.timestamp) = struct.unpack('<Qd', buf)


class BlockSenderChunk:
    '''an incoming chunk packet. This is the main data format'''
    def __init__(self, blockid, size, chunk_id, data, chunk_size, timestamp):
        self.blockid = blockid
        self.size = size
        self.chunk_id = chunk_id
        self.chunk_size = chunk_size
        self.data = data
        self.timestamp = timestamp
        self.format = '<QLHHd'
        self.header_size = struct.calcsize(self.format)
        if data is not None:
            self.packed_size = len(data) + self.header_size
        else:
            self.packed_size = 0
            
    def pack(self):
        '''return a linearized representation'''        
        buf = struct.pack(self.format, self.blockid, self.size, self.chunk_id, self.chunk_size, self.timestamp)
        buf += str(self.data)
        return buf

    def unpack(self, buf):
        '''unpack a linearized representation into the object'''
        (self.blockid, self.size,
         self.chunk_id, self.chunk_size, self.timestamp) = struct.unpack_from(self.format, buf, offset=0)
        self.data = bytearray(buf[self.header_size:])


class BlockSenderBlock:
    '''the state of an incoming or outgoing block'''
    def __init__(self, blockid, size, chunk_size, dest, data=None, callback=None):
        self.blockid = blockid
        self.size = size
        self.chunk_size = chunk_size
        self.num_chunks = int((self.size + (chunk_size-1)) / chunk_size)
        self.acks = BlockSenderSet(blockid, self.num_chunks)
        if data is not None:
            self.data = bytearray(data)
        else:
            self.data = bytearray(size)
        self.timestamp = 0
        self.callback = callback
        self.dest = dest
        self.next_chunk = 0

    def chunk(self, chunk_id):
        '''return data for a chunk'''
        start = chunk_id*self.chunk_size
        return self.data[start:start+self.chunk_size]        

    def complete(self):
        '''return true if all chunks have been sent/received'''
        return self.acks.complete()
        

class BlockSender:
    '''a reliable datagram block sender

    port:          UDP port to listen on
    dest_ip:       default IP to send to
    listen_ip:     IP to listen on (default is wildcard)
    bandwidth:     bandwidth to use in bytes/second (default 100000 bytes/s)
    completed_len: how many completed blocks to remember (default 100)
    chunk_size:    size of data chunks to send in bytes (default 1000)
    backlog:       maximum number of packets to send per tick (default 100)
    rtt:           initial round trip time estimate (0.01 seconds)
    sock:          a optional socket object to use, needs sendto() and recvfrom()
                   plus a fileno() method if recv() with non-zero timeout is used
    debug:         enable debugging (default False)
    '''
    def __init__(self, port, dest_ip=None, listen_ip='', bandwidth=100000,
                 completed_len=1000, chunk_size=1000, backlog=100, rtt=0.01,
                 sock=None,
                 debug=False):
        self.bandwidth = bandwidth
        self.port = port
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((listen_ip, port))
            self.sock.setblocking(False)
        else:
            self.sock = sock
        self.dest_ip = dest_ip
        self.outgoing = []
        self.incoming = []
        self.next_blockid = os.getpid() << 20
        self.last_send_time = time.time()
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
        self.rtt_max = 10

    def set_packet_loss(self, loss):
        '''set a percentage packet loss
        This can be used to simulate lossy networks
        '''
        self.packet_loss = loss

    def send(self, data, dest=None, chunk_size=None, callback=None):
        '''send a data block

        dest:       optional (host,port) tuple
        chunk_size: network send size for this block (defaults to self.chunk_size)
        callback:   optional callback on completion of send
        '''
        if not chunk_size:
            chunk_size = self.chunk_size
        num_chunks = int((len(data) + (chunk_size-1)) / chunk_size)
        if num_chunks > 65535:
            raise BlockSenderException('chunk_size of %u is too small for data length %u' % (chunk_size, len(data)))
        blockid = self.next_blockid
        self.next_blockid += 1
        if dest is None:
            if self.dest_ip is None:
                raise BlockSenderException('no destination specified in send')
            dest = (self.dest_ip, self.port)
        self.outgoing.append(BlockSenderBlock(blockid, len(data), chunk_size, dest, data=data, callback=callback))

    def _debug(self, s):
        '''internal debug function'''
        if self.enable_debug:
            print(s)
        pass

    def _send_object(self, obj, type, dest):
        '''low level object send'''
        if self.packet_loss != 0:
            if random.uniform(0, 1) < self.packet_loss*0.01:
                return
        try:
            buf = obj.pack()
            crc = binascii.crc32(buf)
            buf = struct.pack('<Bl', type, crc) + buf
            self.sock.sendto(buf, dest)
        except socket.error:
            pass
        
    def _send_acks(self):
        '''send extents objects to acknowledge data'''
        if self.acks_needed and self.enable_debug:
            self._debug("sending %u acks" % len(self.acks_needed))
        acks_needed = self.acks_needed.copy()
        for obj in acks_needed:
            try:
                if isinstance(obj, BlockSenderBlock):
                    obj.acks.timestamp = obj.timestamp
                    self._send_object(obj.acks, PKT_ACK, obj.dest)
                else:
                    self._send_object(obj, PKT_COMPLETE, obj.dest)
                self.acks_needed.remove(obj)
            except Exception, msg:
                self._debug(msg)
                return

    def _add_chunk(self, blk, chunk):
        '''add an incoming chunk to a block'''
        blk.acks.add(chunk.chunk_id)
        start = chunk.chunk_id*chunk.chunk_size
        length = len(chunk.data)
        blk.data[start:start+length] = chunk.data
        if blk.acks.complete():
            self.acks_needed.add(BlockSenderComplete(blk.blockid, blk.timestamp, blk.dest))
        else:
            self.acks_needed.add(blk)

    def _check_incoming(self):
        '''check for incoming data or acks. Return True if a packet was received'''
        try:
            (buf, fromaddr) = self.sock.recvfrom(65536)
        except socket.error:
            return False
        if len(buf) == 0:
            return False
        if self.dest_ip is None:
            if self.enable_debug:
                self._debug('connection from %s' % str(fromaddr))
            # setup defaults for send based on first connection
            (self.dest_ip,self.port) = fromaddr
        try:
            if len(buf) < 5:
                self._debug('bad packet %s' % msg)
                return True
            (magic,crc) = struct.unpack_from('<Bl', buf)
            remaining = buf[5:]
            if crc != binascii.crc32(remaining):
                self._debug('bad crc')
                return True                
            if magic == PKT_ACK:
                obj = BlockSenderSet(0,0)
                obj.unpack(remaining)
            elif magic == PKT_COMPLETE:
                obj = BlockSenderComplete(0, None, None)
                obj.unpack(remaining)
            elif magic == PKT_CHUNK:
                obj = BlockSenderChunk(0, 0, 0, "", 0, 0)
                obj.unpack(remaining)
            else:
                self._debug('bad magic %u' % magic)
                return True
        except Exception, msg:
            self._debug('bad packet %s' % msg)
            return True
        tnow = time.time()
        if isinstance(obj, BlockSenderSet):
            # we've received a set of acks for some data
            # find the corresponding outgoing block
            self.rtt_estimate = min(self.rtt_max, 0.95 * self.rtt_estimate + 0.05 * (tnow - obj.timestamp))
            for i in range(len(self.outgoing)):
                out = self.outgoing[i]
                if out.blockid == obj.id:
                    if self.enable_debug:
                        self._debug("ack %s %f" % (str(out.acks), tnow - obj.timestamp))
                    out.acks.update(obj)
                    if out.acks.complete():
                        if self.enable_debug:
                            self._debug("send complete %u" % out.blockid)
                        blk = self.outgoing.pop(i)
                        if blk.callback:
                            blk.callback()
                    return True
            # an ack for something already complete
            return True

        if isinstance(obj, BlockSenderComplete):
            # a full block has been received
            if self.enable_debug:
                self._debug("full ack for blockid %u" % obj.blockid)
            self.rtt_estimate = min(self.rtt_max, 0.95 * self.rtt_estimate + 0.05 * (tnow - obj.timestamp))
            for i in range(len(self.outgoing)):
                out = self.outgoing[i]
                if out.blockid == obj.blockid:
                    if self.enable_debug:
                        self._debug("send complete %u" % out.blockid)
                    blk = self.outgoing.pop(i)
                    if blk.callback:
                        blk.callback()
                    return True
            # an ack for something already complete
            return True

        if isinstance(obj, BlockSenderChunk):
            # we've received a chunk of data
            if self.enable_debug:
                self._debug("got chunk %u of %u" % (obj.chunk_id, obj.blockid))
            if obj.blockid in self.completed:
                # we've already completed this blockid
                self.acks_needed.add(BlockSenderComplete(obj.blockid, obj.timestamp, fromaddr))
                return True
            for i in range(len(self.incoming)):
                blk = self.incoming[i]
                if blk.blockid == obj.blockid:
                    # we have an existing incoming object
                    blk.timestamp = obj.timestamp
                    self._add_chunk(blk, obj)
                    return
            # its a new block
            self.incoming.append(BlockSenderBlock(obj.blockid, obj.size, obj.chunk_size, fromaddr))
            blk = self.incoming[-1]
            blk.timestamp = obj.timestamp
            self._add_chunk(blk, obj)
            return True
        self._debug("unexpected incoming packet type")
        return True


    def available(self, ordered=False):
        '''return the first incoming block if completed or None

        This does no network operations
        '''
        imax = len(self.incoming)
        if ordered:
            imax = 1
        for i in range(imax):
            if self.incoming[i].complete():
                blk = self.incoming.pop(i)
                self.completed.append(blk.blockid)
                while len(self.completed) > self.completed_len:
                    self.completed.pop(0)        
                return blk.data
        return None


    def sendq_size(self):
        '''return number of uncompleted blocks in the send queue'''
        return len(self.outgoing)


    def recv(self, timeout=0, ordered=False):
        '''receive next chunk from network. Return data or None

        timeout:  time to wait for a packet (0 means to return immediately)
        ordered:  return blocks in same order as sent (default False)
        '''
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


    def _send_outgoing(self):
        '''send any outgoing data that is due to be sent'''
        if len(self.outgoing) == 0:
            return

        t = time.time()
        deltat = t - self.last_send_time
        bytes_to_send = int(self.bandwidth * deltat)
        if bytes_to_send <= 0:
            return
        bytes_sent = 0
        chunks_sent = 0

        for blk in self.outgoing:
            if blk.timestamp + 1.2*self.rtt_estimate > t:
                # this block was sent recently, wait for some acks
                continue

            # start where we left off
            chunks = range(blk.next_chunk, blk.num_chunks)
            chunks.extend(range(blk.next_chunk))
                
            for c in chunks:
                if blk.acks.present(c):
                    # we've received an ack for this chunk already
                    continue
                if bytes_sent + blk.chunk_size > bytes_to_send:
                    # this would take us over our bandwidth limit
                    return

                chunk = BlockSenderChunk(blk.blockid, blk.size, c, blk.chunk(c), blk.chunk_size, t)

                if bytes_sent + chunk.packed_size > bytes_to_send:
                    # this would take us over our bandwidth limit
                    return

                bytes_sent += chunk.packed_size
                try:
                    self._send_object(chunk, PKT_CHUNK, blk.dest)
                except Exception, msg:
                    self._debug(msg)
                    return
                blk.next_chunk = (c + 1) % blk.num_chunks
                blk.timestamp = t
                self.last_send_time = t
                chunks_sent += 1
                if chunks_sent == self.backlog:
                    # don't send more than self.backlog per tick
                    return
            
            
    def tick(self, packet_count=None, send_acks=True, send_outgoing=True):
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
            self._send_outgoing()
