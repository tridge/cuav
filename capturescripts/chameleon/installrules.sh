#!/bin/bash

sudo cp ./40-chameleon.rules /etc/udev/rules.d/40-chameleon.rules
sudo chmod 755 /etc/udev/rules.d/40-chameleon.rules
sudo /etc/init.d/udev restart
sudo udevadm control --reload-rules
sudo udevadm trigger

