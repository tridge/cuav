#!/bin/bash

set -e
set -x

screen -S rpi_capture -X quit
screen -S mavproxy -X quit
