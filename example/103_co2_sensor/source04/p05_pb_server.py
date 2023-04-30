#!/bin/python
#|default_exp p05_pb_server
# pip3 install --user protobuf
import time
import socket
import struct
from data_pb2 import Packet, DataRequest, DataResponse
start_time=time.time()
debug=True
_code_git_version="b6b0afaf75d64dac1a914a6eeeb7af646e9dd226"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/103_co2_sensor/source/"
_code_generation_time="00:22:46 of Monday, 2023-05-01 (GMT+1)"
def listen():
    server="192.168.100.122"
    port=12345
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((server,port,))
        s.listen()
        print("{} listening on  server={} port={}".format(((time.time())-(start_time)), server, port), flush=True)
        while (True):
            conn, addr=s.accept()
            with conn:
                data=conn.recv(1024)
                buf=data
                while (data):
                    data=conn.recv(1024)
                    buf += data
                imsg=DataResponse()
                imsg.ParseFromString(buf)
                print("{} nil imsg.index={} imsg.datetime={} imsg.pressure={} imsg.humidity={} imsg.temperature={} imsg.co2_concentration={}".format(((time.time())-(start_time)), imsg.index, imsg.datetime, imsg.pressure, imsg.humidity, imsg.temperature, imsg.co2_concentration), flush=True)
                omsg=DataRequest(start_index=123, count=42).SerializeToString()
                conn.sendall(omsg)
listen()