// no preamble
#define FMT_HEADER_ONLY
#include "core.h"
extern "C" {
#include "driver/uart.h"
#include <esp_log.h>
};

#include "Uart.h"
void Uart::measureCO2(std::deque<Point2D> &fifo) {
  {
    ESP_LOGE(TAG, "measure co2");
    unsigned char command[9] = {0xFF, 0x1, 0x86, 0x0, 0x0, 0x0, 0x0, 0x0, 0x79};
    unsigned char response[9];
    uart_write_bytes(CO2_UART, command, 9);
    auto l = uart_read_bytes(CO2_UART, response, 9, 100);
    if (9 == l) {
      if (((255 == response[0]) && (134 == response[1]))) {
        auto co2 = ((256 * response[2]) + response[3]);
        ESP_LOGE(TAG, "%s", fmt::format("  co2='{}'\n", co2).c_str());
        if (((N_FIFO) - (1)) < fifo.size()) {
          fifo.pop_back();
        }
        auto tv_now = timeval();
        gettimeofday(&tv_now, nullptr);
        auto time_us = (tv_now.tv_sec + ((1.00e-6f) * tv_now.tv_usec));
        auto p = Point2D({.x = time_us, .y = static_cast<double>(co2)});
        fifo.push_front(p);
      }
    }
  }
}
Uart::Uart() {
  ESP_LOGE(TAG, "initialize uart");
  if (uart_is_driver_installed(CO2_UART)) {
    return;
  }
  if (!(ESP_OK == uart_set_pin(CO2_UART, 27, 39, UART_PIN_NO_CHANGE,
                               UART_PIN_NO_CHANGE))) {
    ESP_LOGE(TAG, "error: uart_set_pin 27 39");
  }
  if (!(ESP_OK == uart_driver_install(CO2_UART, 200, 0, 0, nullptr, 0))) {
    ESP_LOGE(TAG, "error: uart_driver_install");
  }
  auto config = uart_config_t({.baud_rate = 9600,
                               .data_bits = UART_DATA_8_BITS,
                               .parity = UART_PARITY_DISABLE,
                               .stop_bits = UART_STOP_BITS_1,
                               .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
                               .source_clk = UART_SCLK_APB});
  if (!(ESP_OK == uart_param_config(CO2_UART, &config))) {
    ESP_LOGE(TAG, "error: uart_param_config");
  }
}
