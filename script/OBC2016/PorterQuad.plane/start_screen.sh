#!/bin/bash

/bin/date >> /root/start_screen.log

export PATH=$PATH:/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin
export HOME=/root
export SHELL=/bin/bash
export TERM=xterm

cd /root
(
  sleep 5
  screen -S MAVProxy -t 0 -d -m -c mav-screenrc
) > screen.log 2>&1
