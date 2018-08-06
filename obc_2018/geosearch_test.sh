#!/bin/bash

# run a test using standard target positions at CMAC for OBC 2018

# target positions taken from test on 5th August 2018
FLAGS="--flag=-35.362669,149.164242,barrell"
FLAGS="$FLAGS --flag=-35.362846,149.164272,barrell"
FLAGS="$FLAGS --flag=-35.363027,149.164295,barrell"

geosearch.py --minscore=500 $FLAGS $*
