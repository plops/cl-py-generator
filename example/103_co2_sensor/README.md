
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
barometric pressure, ambient temperature and gas (VOC). The specific
CO2 sensor would make a great addition to this sensor board.

However, connecting the new sensor was more difficult than I first
thought, as I didn't immediately find pins that I could use for UART,
and the power supply was a challenge. I document my approach here.


# USB-UART converter

The USB-UART converter allows the user to read sensor data from a PC.
The code in source00 shows how to do this in a few lines of Python.

# MCH22 Badge

## UART Connections

The MCH22 has a battery, display, storage, and wifi, but no available
UARTs to connect the sensor. The UART connection between RP2040 and
FPGA can be accessed, but the RP2040 is the USB/serial interface, and
interfering with its firmware could brick the system.

The ESP32 can select arbitrary GPIO pins for three simultaneous
UARTs. Pins greater than gpio34 can only be inputs and can't be used
as UART pins. If a 40MHz baud rate is required, only a subset of
IO_MUX pins will work. Our sensor only requires a 9600 baud
rate. Therefore any pin can be chosen. The following code sets up the
UART:

```
uart_config_t config = {
.baud_rate=115200,
.data_bits = UART_DATA_8_BITS,
.parity = UART_PARITY_DISABLE,
.stop_bits = UART_STOP_BITS_1,
.flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
.source_clk = UART_SCLK_APB,
};

uart_driver_install(UART_NUM_1, 200 /* rx */,0 /* tx */,0,NULL,0);
uart_param_config(UART_NUM_1,&config);
uart_set_pin(UART_NUM_1,GPIO_NUM_4,GPIO_NUM_0,UART_PIN_NO_CHANGE,UART_PIN_NO_CHANGE);

```

# Example program for MCH22 ESP32

I used an example program as a starting point. The example program for
MCH22 ESP32 is named `my_fancy_app_name` and has a size of 1.7GB after
build. The source code is available at
https://github.com/badgeteam/mch2022-template-app. To build the
program, set the IDF_PATH and source the export script, then navigate
to the cmake-build-debug directory and run ninja:


```
export IDF_PATH=~/src/my_fancy_app_name/esp-idf/
source /home/martin/src/my_fancy_app_name/esp-idf/export.sh
cd /home/martin/src/my_fancy_app_name/cmake-build-debug
ninja

```

## Hardware info

Here are some resources for hardware information related to the MCH22 Badge and the ESP32:

    MCH22 Badge pinout: https://badge.team/docs/badges/mch2022/hardware/pinout/
    MCH22 Badge schematic: https://github.com/badgeteam/mch2022-badge-hardware/blob/master/schematic.pdf
    ESP32 pin multiplexing: https://www.esp32.com/viewtopic.php?t=3569
    ESP32 pin layout (WROVER-E module): https://user-images.githubusercontent.com/46353439/116225897-a443dd80-a752-11eb-9437-65b6f20e8e69.png
    ESP32 API reference for GPIO: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/gpio.html
    Building a template app for MCH22 ESP32: https://badge.team/docs/badges/mch2022/software-development/esp-idf/esp_idf_getting_started/

In case the last link is not accessible anymore, it is available on
the waybackmachine: https://web.archive.org/web/20221208223643/http://badge.team/docs/badges/mch2022/software-development/


### Which pins could I use for UART

The MCH22 Badge contains the FPGA, RP2040, and ESP32 and they are all
connected via UART during normal operation (for programming and
logging).

However, the badge's documentatoin doesn't mention any UARTs that can
be used by a user to connect a sensor. There are other UARTs available
on the ESP32 that can be used for our sensor. The ESP32 has three
UARTs, of which only one is configured in the badge firmware. The
other two can be configured to run at almost any pin, but there are
some constraints to keep in mind:

    Not all pins can be configured for input and output.
    Not all pins are routed to the pins of the WROVER-E board (which holds the ESP32 inside a shielding).
    Most, if not all, of the WROVER-E board pins are connected to something on the MCH22 Badge.

Here is a table of available pins and their potential use:

	
| use | gpio   | pin   | function                  | comment                          | decision    |
|-----|--------|-------|---------------------------|----------------------------------|-------------|
|     | GPIO02 | pin24 | SD card data 0            | 45.3k to gnd, 10k to PWR_SDCARD  |             |
|     | GPIO14 | pin13 | SD clock                  | 10k to PWR_SDCARD                |             |
|     | GPIO15 | pin23 | SD command                | 45.3k to gnd, 10k to PWR_SDCARD  |             |
| Y   | GPIO27 | pin12 | SPI chip select fpga (tx) |                                  | could be tx |
|     | GPIO35 | pin7  | SPI MISO fpga             | maybe connected to lcd as well?  | could be tx |
| Y   | GPIO39 | pin5  | Interrupt fpga            | GPIO_NUM >= 34 can only be input | must be rx  |


Note (to self) that the brown cable should go to the TX pin on the sensor.



Initially I thought the SDCard pins are good candidates for my uart
connection because I don't even have a card. They do all have a 10k
resister to PWR_SDCARD, though. And one is even connected to gnd and
PWR_SDCARD.

Therefore I configure GPIO27 and GPIO39 with the second UART. This
seems to work.


For more information on configuring a UART on the ESP32, see the code
snippet provided in the previous section.



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

The serial monitor is an essential tool for reading log messages from
the ESP32 firmware. During initialization of components of the badge,
such as uart or wifi, the serial monitor provides valuable information
to ensure proper configuration. It can also help identify
configuration errors of the uart, e.g. I set the RX buffer too small
at some point.

To open the serial monitor, use the following commands:

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

Using the serial monitor, you can monitor the ESP32 firmware's log
messages and ensure that everything is functioning as expected.

### Notes concerning the power supply


When working with the CO2 sensor, it's important to note that it needs
more than 3.3 volts as a power supply. To solve this, I attached the
VIN of the sensor to the VIN pin located next to the USB connector on
the MCH22 board. However, I did need to disconnect the sensor from the
badge to get the badge booted. Maybe I still have a ground issue.


While the system does keep working when the USB is unplugged and a
battery is connected, the values eventually reach 5000. I believe this
is due to the failure of VIN of the sensor dropping below 3.8 volts.

To solve this issue, I found that running the system with a USB power
brick was effective. However, it's essential to note that no
additional battery should be connected to the badge. If the charge
controller on the badge thinks the battery is full, it won't request
any power from the USB port.
