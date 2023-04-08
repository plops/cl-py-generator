# Reading out an optical CO2 Sensor

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Reading out an optical CO2 Sensor](#reading-out-an-optical-co2-sensor)
    - [Repository Contents](#repository-contents)
    - [Introduction](#introduction)
    - [USB-UART converter](#usb-uart-converter)
    - [MCH22 Badge](#mch22-badge)
        - [UART Connections](#uart-connections)
        - [Example program for MCH22 ESP32](#example-program-for-mch22-esp32)
        - [Hardware info](#hardware-info)
        - [Which pins could I use for UART](#which-pins-could-i-use-for-uart)
        - [How to build and install the ESP32 template software](#how-to-build-and-install-the-esp32-template-software)
        - [Use fmt](#use-fmt)
        - [Serial monitor](#serial-monitor)
        - [Notes concerning the power supply](#notes-concerning-the-power-supply)

<!-- markdown-toc end -->


## Repository Contents ##

| file  | comment                               |   |
|-------|---------------------------------------|---|
| gen00 | python code to read sensor from pc    |   |
| gen01 | c++ code to run on esp32              |   |
| gen02 | c++ code to develop linear ransac fit |   |
| gen03 | c++ code to check continuous update   |   |



## Introduction ##

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


## USB-UART converter ##

The USB-UART converter allows the user to read sensor data from a PC.
The code in source00 shows how to do this in a few lines of Python.

## MCH22 Badge ##

### UART Connections ###

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

### Example program for MCH22 ESP32 ###

I used an example program as a starting point. The example program for
MCH22 ESP32 is named `my_fancy_app_name` and has a size of 1.7GB after
build. The source code is available at
https://github.com/badgeteam/mch2022-template-app.

This repository contains the ESP32 as subrepos. Download like so:
```
cd ~/src
git clone --recursive \
 https://github.com/badgeteam/mch2022-template-app \
 my_fancy_app_name
```

The directory will have a size of 1.5GB, when the clone command is
finished.

Install the SDK:

```
cd ~/src/my_fancy_app_name
cd esp-idf
./install.sh
```


To build the program, set the IDF_PATH and source the export script,
then navigate to the cmake-build-debug directory and run ninja:


```
export IDF_PATH=~/src/my_fancy_app_name/esp-idf/
source /home/martin/src/my_fancy_app_name/esp-idf/export.sh
mkdir /home/martin/src/my_fancy_app_name/cmake-build-debug
cd /home/martin/src/my_fancy_app_name/cmake-build-debug
cmake .. -G Ninja
ninja

```

### Hardware info ###

Here are some resources for hardware information related to the MCH22 Badge and the ESP32:

    MCH22 Badge pinout: https://badge.team/docs/badges/mch2022/hardware/pinout/
    MCH22 Badge schematic: https://github.com/badgeteam/mch2022-badge-hardware/blob/master/schematic.pdf
    ESP32 pin multiplexing: https://www.esp32.com/viewtopic.php?t=3569
    ESP32 pin layout (WROVER-E module): https://user-images.githubusercontent.com/46353439/116225897-a443dd80-a752-11eb-9437-65b6f20e8e69.png
    ESP32 API reference for GPIO: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/gpio.html
    Building a template app for MCH22 ESP32: https://badge.team/docs/badges/mch2022/software-development/esp-idf/esp_idf_getting_started/

In case the last link is not accessible anymore, it is available on
the waybackmachine:
https://web.archive.org/web/20221208223643/http://badge.team/docs/badges/mch2022/software-development/


### Which pins could I use for UART ###

The MCH22 Badge contains the FPGA, RP2040, and ESP32 and they are all
connected via UART during normal operation (for programming and
logging).

However, the badge's documentation doesn't mention any UARTs that can
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
|     | GPIO35 | pin7  | SPI MISO fpga             | maybe connected to LCD as well?  | could be tx |
| Y   | GPIO39 | pin5  | Interrupt fpga            | GPIO_NUM >= 34 can only be input | must be rx  |


Note (to self) that the brown cable should go to the TX pin on the
sensor.



Initially I thought the SDCard pins are good candidates for my uart
connection because I don't even have a card. They do all have a 10k
resister to PWR_SDCARD, though. And one is even connected to gnd and
PWR_SDCARD.

Therefore I configure GPIO27 and GPIO39 with the second UART. This
seems to work.


For more information on configuring a UART on the ESP32, see the code
snippet provided in the previous section.



### How to build and install the ESP32 template software ###

Use `sudo make install` to install the software ion the badge

```
cd ~/src/my_fancy_app_name/b
source ~/src/my_fancy_app_name/esp-idf/export.sh 
```

In the Makefile a tool to flash the binary to the badge is called like
this:

```
cd /home/martin/src/my_fancy_app_name/
sudo python3 tools/webusb_push.py "Template App" build/main.bin --run
```

This works.


### Use fmt ###

The fmtlib library is a useful tool that I use to write log output in
my projects. One of the benefits of using fmtlib is that it allows me
to easily print objects without having to worry about the data
type. This makes it a valuable asset for debugging and
troubleshooting.

To use fmtlib, I downloaded the required header files from the
official fmtlib GitHub repository. This can be done by running the
following commands in the terminal:


```
wget https://raw.githubusercontent.com/fmtlib/fmt/master/include/fmt/core.h
wget https://raw.githubusercontent.com/fmtlib/fmt/master/include/fmt/format.h
wget https://raw.githubusercontent.com/fmtlib/fmt/master/include/fmt/format-inl.h
```

Once downloaded, I included the necessary header files in my project
repository and began using the fmtlib formatting syntax to write log
output. This has helped to simplify my code and make it more readable.


### Serial monitor ###

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

### Notes concerning the power supply ###


When working with the CO2 sensor, it's important to note that it needs
more than 3.3 volts as a power supply. To solve this, I attached the
VIN of the sensor to the VIN pin located next to the USB connector on
the MCH22 board. However, I did need to disconnect the sensor from the
badge to get the badge booted. I am now using the ground pin next to
the prototyping area. Perhaps that is not the same net as the ground
on the pin next to the USB connector.


While the system does keep working when the USB is unplugged and a
battery is connected, the values eventually reach 5000. I believe this
is due to the failure of VIN of the sensor dropping below 3.8
volts. Maybe the light source emits no light anymore and the sensor
thinks the CO2 absorption is very high.

To solve this issue, I found that running the system with a USB power
brick was effective. However, it's essential to note that no
additional battery should be connected to the badge. If the charge
controller on the badge thinks the battery is full, it won't request
any power from the USB port.

## Algorithm

I want to assume CO2 concentration increases linearly and predict when
the concentration will reach 1200 ppm and the room should be aired.

### Ransac


https://www.youtube.com/watch?v=Cu1f6vpEilg
RANSAC - Random Sample Consensus (Cyrill Stachniss)


## Wifi

https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_wifi.html
- i want station mode to connect to an access point with wpa2
- repo with example code that shows how to connect to access point:
 https://github.com/espressif/esp-idf/tree/51772f4fb5c2bbe25b60b4a51d707fa2afd3ac75/examples/wifi/getting_started
- nice: there is a fine timing measurement, that can locate
  devices. but the access point has to support ftm responder mode
  - the example gives an avg raw RTT of 49.218ns. but it doesn't say
    anything about the measurement error.

```
#include "esp_wifi.h"


#define EXAMPLE_ESP_WIFI_SSID      CONFIG_ESP_WIFI_SSID
#define EXAMPLE_ESP_WIFI_PASS      CONFIG_ESP_WIFI_PASSWORD
#define EXAMPLE_ESP_MAXIMUM_RETRY  CONFIG_ESP_MAXIMUM_RETRY

#define ESP_WIFI_SCAN_AUTH_MODE_THRESHOLD WIFI_AUTH_WPA2_PSK


/* FreeRTOS event group to signal when we are connected*/
static EventGroupHandle_t s_wifi_event_group;

/* The event group allows multiple bits for each event, but we only care about two events:
 * - we are connected to the AP with an IP
 * - we failed to connect after the maximum amount of retries */
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1


static const char *TAG = "wifi station";

static int s_retry_num = 0;


static void event_handler(void* arg, esp_event_base_t event_base,
                                int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < EXAMPLE_ESP_MAXIMUM_RETRY) {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGI(TAG, "retry to connect to the AP");
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
        ESP_LOGI(TAG,"connect to the AP fail");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "got ip:" IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

void wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());

    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_got_ip));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = EXAMPLE_ESP_WIFI_SSID,
            .password = EXAMPLE_ESP_WIFI_PASS,
            /* Setting a password implies station will connect to all security modes including WEP/WPA.
             * However these modes are deprecated and not advisable to be used. Incase your Access point
             * doesn't support WPA2, these mode can be enabled by commenting below line */
	     .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA) );
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config) );
    ESP_ERROR_CHECK(esp_wifi_start() );

    ESP_LOGI(TAG, "wifi_init_sta finished.");

    /* Waiting until either the connection is established (WIFI_CONNECTED_BIT) or connection failed for the maximum
     * number of re-tries (WIFI_FAIL_BIT). The bits are set by event_handler() (see above) */
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
            WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
            pdFALSE,
            pdFALSE,
            portMAX_DELAY);

    /* xEventGroupWaitBits() returns the bits before the call returned, hence we can test which event actually
     * happened. */
    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "connected to ap SSID:%s password:%s",
                 EXAMPLE_ESP_WIFI_SSID, EXAMPLE_ESP_WIFI_PASS);
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGI(TAG, "Failed to connect to SSID:%s, password:%s",
                 EXAMPLE_ESP_WIFI_SSID, EXAMPLE_ESP_WIFI_PASS);
    } else {
        ESP_LOGE(TAG, "UNEXPECTED EVENT");
    }

    /* The event will not be processed after unregister */
    ESP_ERROR_CHECK(esp_event_handler_instance_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP,
		instance_got_ip));
    ESP_ERROR_CHECK(esp_event_handler_instance_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID,
		instance_any_id));
    vEventGroupDelete(s_wifi_event_group);
}

void app_main(void)
{
    //Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "ESP_WIFI_MODE_STA");
    wifi_init_sta();
}


```
- https://www.youtube.com/watch?v=_dRrarmQiAM
  - youtube video explaining how to set up wifi on esp32
  - code from video:
    https://github.com/lowlevellearning/esp32-wifi/blob/main/main/wifi-connection.c
  - connects to ap and then to tcp server
  
  
- the video has better code. the code chunk shown above is probably too old

- the esp-idf repo contains
  examples/wifi/getting_started/station/main/station_example_main.c

- wifi fine timing measurement is interesting. i'm not sure yet, if
  the esp32 supports it. the esp-idf docu mentions only esp32-s2 and
  esp32-c3
  - https://www.youtube.com/watch?v=6By78JkCUoo
  - precision 1m .. 2m
