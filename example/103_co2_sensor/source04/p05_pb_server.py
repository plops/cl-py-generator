#!/bin/python
#|default_exp p05_pb_server
# pip3 install --user protobuf
import time
import socket
import struct
from data_pb2 import Packet, DataRequest, DataResponse
start_time=time.time()
debug=True
_code_git_version="17ae302b260d342c0bf28a020136eac321b9b851"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/103_co2_sensor/source/"
_code_generation_time="00:01:35 of Saturday, 2023-04-22 (GMT+1)"
def listen():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("192.168.2.122",12345,))
        s.listen()
        print("{} listening on localhost:12345 ".format(((time.time())-(start_time))))
        while (True):
            conn, addr=s.accept()
            with conn:
                data=conn.recv(1024)
                buf=data
                while (data):
                    data=conn.recv(1024)
                    buf += data
                imsg=DataResponse()
                print("{} nil imsg.index={} imsg.datetime={} imsg.pressure={} imsg.humidity={} imsg.temperature={} imsg.co2_concentration={}".format(((time.time())-(start_time)), imsg.index, imsg.datetime, imsg.pressure, imsg.humidity, imsg.temperature, imsg.co2_concentration))
                omsg=DataRequest(start_index=123, count=42).SerializeToString()
                conn.sendall(omsg)
listen()