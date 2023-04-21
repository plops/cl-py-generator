#!/bin/python
#|default_exp p05_pb_server
# pip3 install --user protobuf
import time
import socket
import struct
from data_pb2 import Packet, DataRequest, DataResponse
start_time=time.time()
debug=True
_code_git_version="d50b133f2dd85de9981613ef3ea75087d872ddf3"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/103_co2_sensor/source/"
_code_generation_time="08:39:01 of Friday, 2023-04-21 (GMT+1)"
def listen():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost",12345,))
        s.listen()
        print("{} listening on localhost:12345 ".format(((time.time())-(start_time))))
        while (True):
            conn, addr=s.accept()
            with conn:
                print("{} connection addr={}".format(((time.time())-(start_time)), addr))
                print("{} wait for DataResponse message ".format(((time.time())-(start_time))))
                data=conn.recv(1024)
                buf=data
                while (data):
                    data=conn.recv(1024)
                    buf += data
                print("{} finished reading buf={}".format(((time.time())-(start_time)), buf))
                imsg=DataResponse()
                print("{} nil imsg.ParseFromString(buf)={}".format(((time.time())-(start_time)), imsg.ParseFromString(buf)))
                print("{} nil imsg.index={} imsg.datetime={} imsg.pressure={} imsg.humidity={} imsg.temperature={} imsg.co2_concentration={}".format(((time.time())-(start_time)), imsg.index, imsg.datetime, imsg.pressure, imsg.humidity, imsg.temperature, imsg.co2_concentration))
                print("{} send DataRequest message ".format(((time.time())-(start_time))))
                omsg=DataRequest(start_index=123, count=42).SerializeToString()
                conn.sendall(omsg)
                print("{} connection closed ".format(((time.time())-(start_time))))
listen()