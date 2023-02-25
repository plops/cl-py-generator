
|	|	||	
|  gen00 | python code to read sensor from pc |
| gen01 |  c++ code to run on esp32
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

looks like the badge website just disappeared. it is on the internet archive:

https://web.archive.org/web/20221208223643/http://badge.team/docs/badges/mch2022/software-development/


### Which pins could I use for uart

.|GPIO02 | pin24  | SD card data 0 |  45.3k to gnd, 10k to PWR_SDCARD
.|GPIO14 | pin13| SD clock | 10k to PWR_SDCARD 
.|GPIO15 | pin23 | SD command | 45.3k to gnd, 10k to PWR_SDCARD
Y|GPIO27| pin12 | SPI chip select fpga
.|GPIO35 | pin7| SPI MISO fpga
Y|GPIO39 | pin5 |	   Interrupt fpga


I think the SDCard pins are good because I don't even have one. They
do all have a 10k resister to PWR_SDCARD, though. And one is even
connected to gnd and PWR_SDCARD.

So maybe I can get away with the FPGA links. Hopefully, they are not
used.

Use `sudo make install` to install the software ion the badge

```
cd ~/src/my_fancy_app_name/b
source ~/src/my_fancy_app_name/esp-idf/export.sh 



```


If I run this, then I get a long text "Flashing attempt detected"
```

/home/martin/.espressif/python_env/idf4.4_py3.10_env/bin/python esp-idf/components/esptool_py/esptool/esptool.py -p /dev/ttyACM0 -b 460800 --before default_reset --after hard_reset --chip esp32  write_flash --flash_mode dio --flash_size detect --flash_freq 80m 0x1000 build/bootloader/bootloader.bin 0x8000 build/partition_table/partition-table.bin 0xf000 build/phy_init_data.bin 0x10000 build/main.bin

```

It says I should install the app using webusb_push.py tool.

In the Makefile it is called like this:

```
cd /home/martin/src/my_fancy_app_name/
sudo python3 tools/webusb_push.py "Template App" build/main.bin --run

```

This works.