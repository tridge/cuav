#!/bin/bash

set -e
set -x

screen -S image_capture -X quit
screen -S mavproxy -X quit
