#!/usr/bin/env python

import sys, os, subprocess, time, fnmatch

# script for main GCS laptop to keep network setup correctly

TRIDGELLNET="203.217.61.45"
OZLABSORG="203.11.71.1"
PI_TELSTRA="192.168.3.10"
PI_OPTUS="192.168.3.11"

def run_command(cmd):
    '''run a shell command'''
    argv = cmd.split()
    try:
        ret = subprocess.check_output(argv, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return False
    return True

def command_find_string(cmd, s):
    '''return True if a command output contains the given string or list of strings on one line'''
    argv = cmd.split()
    try:
        ret = subprocess.check_output(argv, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return False
    if not isinstance(s, list):
        s = [s]

    lines = ret.splitlines()
    for line in lines:
        missing = False
        for s2 in s:
            if line.find(s2) == -1:
                missing = True
                break
        if not missing:
            return True
    return False

def ping_check(host):
    '''check if a host is up'''
    return run_command("ping -n -W1 -q -c2 %s" % host)

def ping_check_remote(host_gw, host):
    '''check if a host is up via ssh'''
    return run_command("ssh -o ConnectTimeout=2 pi@%s ping -n -W1 -q -c2 %s" % (host_gw, host))

while True:
    print(time.asctime())

    pi_telstra_ok = ping_check(PI_TELSTRA)
    pi_optus_ok = ping_check(PI_OPTUS)

    if pi_telstra_ok:
        pi_telstra_gw_ok = ping_check_remote(PI_TELSTRA, TRIDGELLNET)
    else:
        pi_telstra_gw_ok = False

    if pi_optus_ok:
        pi_optus_gw_ok = ping_check_remote(PI_OPTUS, OZLABSORG)
    else:
        pi_optus_gw_ok = False

    print("Telstra: %s" % pi_telstra_ok)
    print("Optus: %s" % pi_optus_ok)
    print("TelstraGW: %s" % pi_telstra_gw_ok)
    print("OptusGW: %s" % pi_optus_gw_ok)

    if pi_telstra_gw_ok:
        if not command_find_string("route -n", [TRIDGELLNET, PI_TELSTRA]):
            print("Adding tridgell.net route via pi-telstra")
            run_command("sudo route del -host %s dev wlan0" % TRIDGELLNET)
            run_command("sudo route add -host %s gw %s dev eth2" % (TRIDGELLNET, PI_TELSTRA))
    else:
        if not command_find_string("route -n", [TRIDGELLNET, "wlan0"]):
            print("Adding tridgell.net route via wlan0")
            run_command("sudo route del -host %s gw %s dev eth2" % (TRIDGELLNET, PI_TELSTRA))
            run_command("sudo route add -host %s dev wlan0" % TRIDGELLNET)

    if pi_optus_gw_ok:
        if not command_find_string("route -n", [OZLABSORG, PI_OPTUS]):
            print("Adding ozlabs.org route via pi-optus")
            run_command("sudo route del -host %s dev wlan0" % OZLABSORG)
            run_command("sudo route add -host %s gw %s dev eth2" % (OZLABSORG, PI_OPTUS))
    else:
        if not command_find_string("route -n", [OZLABSORG, "wlan0"]):
            print("Adding ozlabs.org route via wlan0")
            run_command("sudo route del -host %s gw %s dev eth2" % (OZLABSORG, PI_OPTUS))
            run_command("sudo route add -host %s dev wlan0" % OZLABSORG)
            
    time.sleep(5)

