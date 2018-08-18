#!/bin/bash

# need to run from home directory
cd ~/

## Raspi-Config
sudo raspi-config nonint do_expand_rootfs
sudo raspi-config nonint do_camera 0
sudo raspi-config nonint do_ssh 0
sudo raspi-config nonint do_serial 1
sudo raspi-config nonint do_wifi_country AU

## Disable Bluetooth, so we have serial port access
echo "" | sudo tee -a /boot/config.txt >/dev/null
echo "# Disable Bluetooth" | sudo tee -a /boot/config.txt >/dev/null
echo "dtoverlay=pi3-disable-bt" | sudo tee -a /boot/config.txt >/dev/null

sudo systemctl disable hciuart.service
sudo systemctl disable bluealsa.service
sudo systemctl disable bluetooth.service

## Update everything
sudo apt update
sudo apt upgrade -y

## Add required packages for cuav and MAVProxy
sudo apt install git screen python python-dev libxml2-dev libxslt-dev python-pip python-matplotlib -y
sudo apt install python-opencv libjpeg62-turbo-dev rdate etckeeper -y
pip install pyserial future --user
pip install pymavlink pytest pytest-mock --user

## Ensure the ~/.local/bin is on the system path
echo "PATH=\$PATH:~/.local/bin" >> ~/.profile
source ~/.profile

## Git clone MAVProxy and cuav

[ -d mavproxy ] || {
    git clone https://github.com/ardupilot/mavproxy.git
}

## and build them
pushd ~/mavproxy
python setup.py build install --user
popd

[ -d cuav ] || {
    git clone https://github.com/canberrauav/cuav.git
}

pushd ~/cuav
python setup.py build install --user
popd

## Install zerotier (https://www.zerotier.com/download.shtml)
curl -s 'https://pgp.mit.edu/pks/lookup?op=get&search=0x1657198823E52A61' | gpg --import && \
if z=$(curl -s 'https://install.zerotier.com/' | gpg); then echo "$z" | sudo bash; fi

## Install a package that will automatically mount & unmount USB drives
sudo apt install usbmount -y

## Setup wifi so you can connect to a secured network
#wpa_passphrase "MyWiFi" "MyPassphrase" | sudo tee -a /etc/wpa_supplicant/wpa_supplicant.conf

## Add your SSH pub key
(umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys)
chown -R $(id -u pi):$(id -g pi) ~/.ssh
curl -sL https://github.com/tridge.keys >> ~/.ssh/authorized_keys
curl -sL https://github.com/stephendade.keys >> ~/.ssh/authorized_keys

## Remove any not required packages
sudo apt autoremove -y

## For security, disable ssh login using username/passwords (ssh keys only!)
sudo sed -i 's/^\#PasswordAuthentication yes/PasswordAuthentication no/g' /etc/ssh/sshd_config

## fake completing the raspi-config
sed '/do_finish()/,/^$/!d' /usr/bin/raspi-config | sed -e '1i ASK_TO_REBOOT=0;' -e '$a do_finish' | bash

## Join zerotier network
#sudo zerotier-cli join xxxxx

## And last of all, reboot
sudo reboot

