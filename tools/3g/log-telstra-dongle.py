#!/usr/bin/python

import time
import requests
import json
import csv
import sys
import datetime

class TelstraDongleLogger():
    def __init__(self, fields):
        self.ip = "192.168.0.1"
        self.script = "goform/goform_get_cmd_process"
        self.fields = fields

    def url(self):
        fieldstring = ",".join(self.fields);
        return "http://{ip}/{script}?multi_data=1&isTest=false&cmd={fieldstring}&_=1475535303952".format(ip=self.ip, script=self.script, fieldstring=fieldstring)

    def poll(self):
        r = requests.get(self.url())
        if r.status_code != 200:
            raise ValueError("Bad response from request (%d)" % r.status_code)
        stuff = json.loads(r.content)
        return stuff

if __name__ == '__main__':
    fields = ['signalbar','network_type']
    csv_writer = csv.writer(sys.stdout)
    headings = ["timestamp"]
    headings.extend(fields)
    csv_writer.writerow(headings)
    dongle = TelstraDongleLogger(fields)
    while True:
        results = dongle.poll()
        row = [int(time.time())*1000000+datetime.datetime.now().microsecond]
        for field in fields:
            row.append(results[field])
        csv_writer.writerow(row)

        time.sleep(1)
