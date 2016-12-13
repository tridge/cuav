#!/bin/bash
#
# The purpose of this script is to guess what "dotfile" is run
# by your local shell to configure it's environment
#
# it only works for bash

if [ -n "$BASH_VERSION" ]; then
    if [ -f "$HOME/.bash_profile" ]; then
        PROFILE_FILE="$HOME/.bash_profile"
    elif [ -f "$HOME/.bash_login" ]; then
        PROFILE_FILE="$HOME/.bash_login"
    else
        PROFILE_FILE="$HOME/.profile"
    fi
    echo $PROFILE_FILE
    exit 0
else
    # sorry, I'm to stupid for your shell 
    exit 1
fi
