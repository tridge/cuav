#!/bin/bash

set -e
set -x

screen -S chameleon_capture -X quit
screen -S mavproxy -X quit
