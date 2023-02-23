
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