
| file  | comment                               |   |
|-------|---------------------------------------|---|
| gen00 | python code to read sensor from pc    |   |
| gen01 | c++ code to run on esp32              |   |
| gen02 | c++ code to develop linear ransac fit |   |

# Introduction

It's useful to read CO2 levels for a variety of reasons, such as
monitoring indoor air quality, ensuring proper ventilation in
buildings, and measuring the effectiveness of carbon capture and
storage systems. There are different types of CO2 sensors available in
the market, including those that measure absorption. These sensors are
good because they are only sensitive to CO2, making them more
specific. One example of a good candidate is the MH-Z19B sensor.

However, it's worth noting that another class of sensors that are
sometimes marketed for CO2 detection actually detect volatile organic
compounds. These sensors use adsorption or absorption of the gas in a
sensitive surface of a semiconductor and perform electrical
measurements, and while they are cheaper than optical sensors, they
have much less specificity.

When it comes to purchasing CO2 sensors, it can be challenging to
determine which sensor is used in fully finished CO2 modules with
batteries and displays on websites like AliExpress. I opted to buy
standalone sensors instead, like the MH-Z19B, and attach them to a
device of my choice.

Initially I tried to read the sensor using a USB-to-UART converter
that allowed me to measure with a laptop, but ultimately I want a
standalone device.

I have a badge left from MCH22 and decided to attach the sensor to
it. It has display, battery, USB- and Wifi connectivity, storage,
several microprocessors and even some sensors from Bosch (BNO055,
BME680).  The BE680 is a Gas sensor measuring relative humidity,
barometric pressure, ambient temperature and gas (VOC).


However, connecting the new sensor was more difficult than I first
thought, as I didn't immediately find pins that I could use for UART,
and the power supply was a challenge. I document my approach here.


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

There someone uses (uart1_tx=gpio27, uart_rx=gpio14) and
(uart2_tx=gpio2,uart_rx=gpio15) on the WROVER-E module.

Pins > gpio34 can only be inputs and can't be used as UART pins.

If you need 40MHz baudrate then only a subset of IO_MUX pins work. I
need 9600 baud so I think I can choose any pin.

```
uart_config_t config = {
.baud_rate=115200,
.data_bits = UART_DATA_8_BITS,
.parity = UART_PARITY_DISABLE,
.stop_bits = UART_STOP_BITS_1,
.flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
.source_clk = UART_SCLK_APB,
};

uart_driver_install(UART_NUM_1, BUF_SIZE*2,0,0,NULL,0);
uart_param_config(UART_NUM_1,&config);
uart_set_pin(UART_NUM_1,GPIO_NUM_4,GPIO_NUM_0,UART_PIN_NO_CHANGE,UART_PIN_NO_CHANGE);

```

# Example program for MCH22 ESP32

my_fancy_app_name 1.7GB after build

```
https://github.com/badgeteam/mch2022-template-app

```

```
export IDF_PATH=~/src/my_fancy_app_name/esp-idf/
source /home/martin/src/my_fancy_app_name/esp-idf/export.sh
cd /home/martin/src/my_fancy_app_name/cmake-build-debug
ninja

```

## Hardware info

https://badge.team/docs/badges/mch2022/hardware/pinout/
https://github.com/espressif/esp-idf/issues/6939 uarts available

https://user-images.githubusercontent.com/46353439/116225897-a443dd80-a752-11eb-9437-65b6f20e8e69.png pin layout esp32-wover

https://www.esp32.com/viewtopic.php?t=3569 pin multiplexing on esp32

https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/gpio.html

https://badge.team/docs/badges/mch2022/software-development/esp-idf/esp_idf_getting_started/ building template app


https://badge.team/docs/badges/mch2022/hardware/pinout/ mch22 badge pinout
https://github.com/badgeteam/mch2022-badge-hardware/blob/master/schematic.pdf schematic of mch22

looks like the badge website just disappeared. it is on the internet archive (waybackmachine):

https://web.archive.org/web/20221208223643/http://badge.team/docs/badges/mch2022/software-development/


### Which pins could I use for uart

The mch22 badge documentation doesn't mention any uarts for me to
use. While fpga, rp2040 and esp32 are connected with uarts, these
channels are used during normal operation (for programming and
logging).


The ESP32 contains 3 uarts, of which one is configured. The others can
be configured to run at almost any pin. There are however constraints
because:
- not all pins can be configured for input and output
- not all pins are routed to the pins of the wrover-e board (that
  holds the esp32 inside a shielding)
- most (all?) of the wrover-e board pins are connected to something on
  the mch22 badge
	
| use | gpio   | pin   | function                  | comment                          | decision    |
|-----|--------|-------|---------------------------|----------------------------------|-------------|
| .   | GPIO02 | pin24 | SD card data 0            | 45.3k to gnd, 10k to PWR_SDCARD  |             |
| .   | GPIO14 | pin13 | SD clock                  | 10k to PWR_SDCARD                |             |
| .   | GPIO15 | pin23 | SD command                | 45.3k to gnd, 10k to PWR_SDCARD  |             |
| Y   | GPIO27 | pin12 | SPI chip select fpga (tx) |                                  | could be tx |
| .   | GPIO35 | pin7  | SPI MISO fpga             | maybe connected to lcd as well?  | could be tx |
| Y   | GPIO39 | pin5  | Interrupt fpga            | GPIO_NUM >= 34 can only be input | must be rx  |
|     |        |       |                           |                                  |             |


- the brown cable should go to tx on the sensor


I think the SDCard pins are good candidates for my uart connection
because I don't even have a card. They do all have a 10k resister to
PWR_SDCARD, though. And one is even connected to gnd and PWR_SDCARD.

So maybe I can get away with the FPGA links. Hopefully, they are not
used.

- Note: I configure GPIO27 and GPIO39 with the second UART. This seems
  to work.


### How to build and install the ESP32 template software

Use `sudo make install` to install the software ion the badge

```
cd ~/src/my_fancy_app_name/b
source ~/src/my_fancy_app_name/esp-idf/export.sh 
```

In the Makefile a tool to flash the binary to the badge is called like this:

```
cd /home/martin/src/my_fancy_app_name/
sudo python3 tools/webusb_push.py "Template App" build/main.bin --run
```

This works.


### Use fmt

```
wget https://raw.githubusercontent.com/fmtlib/fmt/master/include/fmt/core.h
wget https://raw.githubusercontent.com/fmtlib/fmt/master/include/fmt/format.h
wget https://raw.githubusercontent.com/fmtlib/fmt/master/include/fmt/format-inl.h
```


### Serial monitor

```
make monitor
```
or

```
source "$(IDF_PATH)/export.sh" && idf.py monitor -p /dev/ttyACM0
```
or

```
/home/martin/.espressif/python_env/idf4.4_py3.10_env/bin/python /home/martin/src/my_fancy_app_name/esp-idf/tools/idf_monitor.py -p /dev/ttyACM0 -b 115200 --toolchain-prefix xtensa-esp32-elf- --target esp32 --revision 3 /home/martin/src/my_fancy_app_name/build/main.elf -m '/home/martin/.espressif/python_env/idf4.4_py3.10_env/bin/python' '/home/martin/src/my_fancy_app_name/esp-idf/tools/idf.py'
```

### Notes concerning the power supply

- note that the sensor needs more than 3.3v as power supply i attached
- vin of sensor to the vin pin that is next to the usb connector i
- need to disconnect to start board though. maybe i will also need to
- use the ground from there

- while the system keeps working when usb is unplugged and a battery
  is connected, the values will eventually go to 5000. i think this is
  a failure when vin of the sensor drops below 3.8v.

- i can run the system with a usb power brick. however, it is
  important that no additional battery is connected to the badge. i
  think if the charge controller on the badge thinks the battery is
  full, it will not request any power from the usb port.
a
