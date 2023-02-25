#include <cmath>
#include <deque>
#include <random>
#include <vector>
const int N_FIFO = 240;
const int RANSAC_MAX_ITERATIONS = 100;
const float RANSAC_INLIER_THRESHOLD = 0.1;
const int RANSAC_MIN_INLIERS = 50;
std::deque<unsigned short> fifo(N_FIFO, 0);
struct Point2D {
  double x;
  double y;
};
typedef struct Point2D Point2D;

double distance(Point2D p, double m, double b) {
  return ((abs(((p.y) - (((m * p.x) + b))))) / (sqrt((1 + (m * m)))));
}

void ransac_line_fit(std::deque<Point2D> &data, double &m, double &b) {
  std::random_device rd;
  auto gen = std::mt19937(rd());
  auto distrib = std::uniform_int_distribution<>(0, ((data.size()) - (1)));
  auto best_inliers = std::vector<Point2D>();
  auto best_m = (0.);
  auto best_b = (0.);
  for (auto i = 0; i < RANSAC_MAX_ITERATIONS; i += 1) {
    auto idx1 = distrib(gen);
    auto idx2 = distrib(gen);
    auto p1 = data[idx1];
    auto p2 = data[idx2];
    auto m = ((((p2.y) - (p1.y))) / (((p2.x) - (p1.x))));
    auto b = ((p1.y) - ((m * p1.x)));
    auto inliers = std::vector<Point2D>();
    for (auto &p : data) {
      if (distance(p, m, b) < RANSAC_INLIER_THRESHOLD) {
        inliers.push_back(p);
      }
    };
    if (RANSAC_MIN_INLIERS < inliers.size()) {
      auto sum_x = (0.);
      auto sum_y = (0.);
      for (auto &p : inliers) {
        sum_x += p.x;
        sum_y += p.y;
      };
      auto avg_x = ((sum_x) / (inliers.size()));
      auto avg_y = ((sum_y) / (inliers.size()));
      auto var_x = (0.);
      auto cov_xy = (0.);
      for (auto &p : inliers) {
        var_x += (((p.x) - (avg_x)) * ((p.x) - (avg_x)));
        cov_xy += (((p.x) - (avg_x)) * ((p.y) - (avg_y)));
      };
      auto m = ((cov_xy) / (var_x));
      auto b = ((avg_y) - ((m * avg_x)));
      if (best_inliers.size() < inliers.size()) {
        best_inliers = inliers;
        best_m = m;
        best_b = b;
      }
    }
  }
  m = best_m;
  b = best_b;
}
extern "C" {
#include "driver/uart.h"
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

#define CO2_UART UART_NUM_2
#define BUF_SIZE 100

void uart_init() {
  if (uart_is_driver_installed(CO2_UART)) {
    return;
  }
  uart_set_pin(CO2_UART, 4, 5, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
  uart_driver_install(CO2_UART, (BUF_SIZE * 2), 0, 0, nullptr, 0);
  auto config = uart_config_t({.baud_rate = 9600,
                               .data_bits = UART_DATA_8_BITS,
                               .parity = UART_PARITY_DISABLE,
                               .stop_bits = UART_STOP_BITS_1,
                               .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
                               .rx_flow_ctrl_thresh = 0,
                               .source_clk = UART_SCLK_APB});
  ESP_ERROR_CHECK(uart_param_config(CO2_UART, &config));
}

void measureCO2() {
  {
    unsigned char command[9] = {0xFF, 0x1, 0x86, 0x0, 0x0, 0x0, 0x0, 0x0, 0x79};
    unsigned char response[9];
    uart_write_bytes(CO2_UART, command, 9);
    auto l = uart_read_bytes(CO2_UART, response, 9, 100);
    if (9 == l) {
      if (((255 == response[0]) && (134 == response[1]))) {
        auto co2 = ((256 * response[2]) + response[3]);
        if (N_FIFO < fifo.size()) {
          fifo.pop_back();
        }
        fifo.push_front(co2);
      }
    }
  }
}

void drawCO2(pax_buf_t *buf) {
  auto hue = 12;
  auto sat = 255;
  auto bright = 255;
  auto col = pax_col_hsv(hue, sat, bright);
  for (auto i = 0; i < ((fifo.size()) - (1)); i += 1) {
    pax_draw_line(buf, col, i, fifo[i], (i + 1), fifo[(i + 1)]);
  }
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
    measureCO2();
    auto hue = ((esp_random()) & (255));
    auto sat = 255;
    auto bright = 255;
    auto col = pax_col_hsv(hue, sat, bright);
    pax_background(&buf, col);
    auto text = "hello martin";
    auto font = pax_font_saira_condensed;
    auto dims = pax_text_size(font, font->default_size, text);
    drawCO2(&buf);
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
