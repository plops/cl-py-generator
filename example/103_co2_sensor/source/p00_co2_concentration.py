#|default_exp p00_co2_concentration
import os
import time
import serial
import datetime
start_time=time.time()
debug=True
_code_git_version="0b1083c19076d8d4c24b695d8c231cad9e16a2f4"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/103_co2_sensor/source/"
_code_generation_time="20:33:53 of Thursday, 2023-02-16 (GMT+1)"
# https://www.winsen-sensor.com/d/files/PDF/Infrared%20Gas%20Sensor/NDIR%20CO2%20SENSOR/MH-Z19%20CO2%20Ver1.0.pdf
# sudo apt install python3-serial
ser=serial.Serial("/dev/ttyUSB0", 9600)
while (True):
    ser.write(b"\xff\x01\x86\x00\x00\x00\x00\x00\x79")
    data=ser.read(9)
    if ( ((((0xff)==(data[0]))) and (((0x86)==(data[1])))) ):
        co2=((((256)*(data[2])))+(data[3]))
        print("{} vars co2={}".format(((time.time())-(start_time)), co2))
    time.sleep((0.50    ))