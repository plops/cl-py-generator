
# USB-UART converter

I can read the sensor using a USB converter.

# MCH22 Badge

I would like to connect the sensor to the MCH22 (which has battery,
display, storage and wifi). Unfortunately, no obvious UARTs are free.


I could access the UART connection between RP2040 and FPGA. There are
testpoints for RX and TX. But that the RP2040 is the USB/serial
interface and I could brick the system if I mess up its firmware.


The ESP32 can select arbitrary GPIO pins for 3 simultaneous UARTS:
https://github.com/espressif/esp-idf/issues/6939