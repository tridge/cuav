#!/bin/bash
#
# The purpose of this script is to ensure ~/.local/bin is on the path
#
# This script should be idempotent, and it should have no effect
# if ~/.local/bin is already on you path. It assumes your shell is
# bash, but should have no effect if you are using another shell.

DOT_LOCAL_BIN="$HOME/.local/bin"
if [ -n "$BASH_VERSION" ]; then
    PROFILE_FILE=`./guess_dotfile.sh`
    FILTER_PATH=`echo $PATH | grep $DOT_LOCAL_BIN`
    if [ -n "$FILTER_PATH" ]; then
	echo "$DOT_LOCAL_BIN is already on the path, no change required"
    else
	echo "$DOT_LOCAL_BIN not on path"
	PFMOD=`cat $PROFILE_FILE | grep 'export PATH' | grep "$DOT_LOCAL_BIN"`
	DOTP_MSG="please run '. $PROFILE_FILE' to fix your PATH"
	DOTP_MSG="$DOTP_MSG for the remainder of this shell session"
	if [ -n "$PFMOD" ]; then
	    echo "$PROFILE_FILE already adds $DOT_LOCAL_BIN to PATH"
	    echo $DOTP_MSG
	    echo "if that doesn't work, please check the file manually"
	else
	    echo "$PROFILE_FILE does not seem to add $DOT_LOCAL_BIN to PATH"
            echo "editing $PROFILE_FILE to add it"
            COMMENT="# cuav programs installed with 'setup.py install --local'"
	    echo $COMMENT >> $PROFILE_FILE
            echo "export PATH=$PATH:$DOT_LOCAL_BIN" >> $PROFILE_FILE
	    echo $DOTP_MSG
	fi
    fi
fi
