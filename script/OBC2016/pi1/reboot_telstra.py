#!/usr/bin/env python

import pexpect, sys

IP="192.168.0.1"

t = pexpect.spawn('telnet 192.168.0.1', logfile=sys.stdout)
t.expect('login')
t.send('root\n')
t.expect('Password')
t.send('zte9x15\n')
t.expect('root@')
t.send('reboot\n')
t.expect('going down for reboot')
