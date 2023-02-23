extern "C" {
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "hardware.h"
#include "ili9341.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "pax_codecs.h"
#include "pax_gfx.h"
#include "soc/rtc.h"
#include "soc/rtc_cntl_reg.h"
#include "wifi_connect.h"
#include "wifi_connection.h"
#include <esp_log.h>

static const char *TAG = "mch2022-co2-app";
static pax_buf_t buf;
xQueueHandle buttonQueue;

void disp_flush() {
  ili9341_write(get_ili9341(), static_cast<const uint8_t *>(buf.buf));
}

void exit_to_launcher() {
  REG_WRITE(RTC_CNTL_STORE0_REG, 0);
  esp_restart();
}

void uart_init() {
  auto BUF_SIZE = 1024;
  auto config = uart_config_t({.baud_rate = 9600,
                               .data_bits = UART_DATA_8_BITS,
                               .parity = UART_PARITY_DISABLE,
                               .stop_bits = UART_STOP_BITS_1,
                               .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
                               .source_clk = UART_SCLK_APB});
  uart_driver_install(UART_NUM_1, (BUF_SIZE * 2), 0, 0, nullptr, 0);
  uart_param_config(UART_NUM_1, &config);
  uart_set_pin(UART_NUM_1, GPIO_NUM_4, GPIO_NUM_0, UART_PIN_NO_CHANGE,
               UART_PIN_NO_CHANGE);
}

void app_main() {
  ESP_LOGI(TAG, "welcome to the template app");
  bsp_init();
  bsp_rp2040_init();
  buttonQueue = get_rp2040()->queue;

  pax_buf_init(&buf, nullptr, 320, 240, PAX_BUF_16_565RGB);
  nvs_flash_init();
  wifi_init();
  uart_init();
  while (1) {
    auto hue = ((esp_random()) & (255));
    auto sat = 255;
    auto bright = 255;
    auto col = pax_col_hsv(hue, sat, bright);
    pax_background(&buf, col);
    auto text = "hello martin";
    auto font = pax_font_saira_condensed;
    auto dims = pax_text_size(font, font->default_size, text);
    pax_draw_text(&buf, 0xFF000000, font, font->default_size,
                  ((((buf.width) - (dims.x))) / ((2.0f))),
                  ((((buf.height) - (dims.y))) / ((2.0f))), text);
    disp_flush();
    auto message = rp2040_input_message_t();
    xQueueReceive(buttonQueue, &message, portMAX_DELAY);
    if (((RP2040_INPUT_BUTTON_HOME == message.input) && (message.state))) {
      exit_to_launcher();
    }
  }
}
};
