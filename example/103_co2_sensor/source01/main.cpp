#define FMT_HEADER_ONLY
#include "core.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <random>
#include <vector>
const int N_FIFO = 320;
const int RANSAC_MAX_ITERATIONS = 320;
const float RANSAC_INLIER_THRESHOLD = 5.0;
const int RANSAC_MIN_INLIERS = 2;
struct Point2D {
  double x;
  double y;
};
typedef struct Point2D Point2D;

struct PointBME {
  double x;
  double temperature;
  double humidity;
  double pressure;
};
typedef struct PointBME PointBME;

std::deque<Point2D> fifo;
std::deque<PointBME> fifoBME;
extern "C" {
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "nvs_flash.h"
static EventGroupHandle_t s_wifi_event_group;
// event group should allow two different events
// 1) we are connected to access point with an ip
// 2) we failed to connect after a maximum amount of retries

static int s_retry_num = 0;
};
enum {
  WIFI_CONNECTED_BIT = BIT0,
  WIFI_FAIL_BIT = BIT1,
  EXAMPLE_ESP_MAXIMUM_RETRY = 7
};

void event_handler(void *arg, esp_event_base_t event_base, int32_t event_id,
                   void *event_data) {
  if (((WIFI_EVENT == event_base) && (WIFI_EVENT_STA_START == event_id))) {
    esp_wifi_connect();
  } else {
    if (((WIFI_EVENT == event_base) &&
         (WIFI_EVENT_STA_DISCONNECTED == event_id))) {
      if (s_retry_num < EXAMPLE_ESP_MAXIMUM_RETRY) {
        esp_wifi_connect();
        s_retry_num++;
        fmt::print("retry to connect to the access point\n");

      } else {
        xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
      }
      fmt::print("connection to the access point failed\n");

    } else {
      if (((IP_EVENT == event_base) && (IP_EVENT_STA_GOT_IP == event_id))) {
        auto event = static_cast<ip_event_got_ip_t *>(event_data);
        fmt::print("got ip:  IP2STR(&event->ip_info.ip)='{}'\n",
                   IP2STR(&event->ip_info.ip));
        s_retry_num = 0;

        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
      }
    }
  }
}

void wifi_init_sta() {
  s_wifi_event_group = xEventGroupCreate();

  ESP_ERROR_CHECK(esp_netif_init());
}

double distance(Point2D p, double m, double b) {
  return ((abs(((p.y) - (((m * p.x) + b))))) / (sqrt((1 + (m * m)))));
}

void ransac_line_fit(std::deque<Point2D> &data, double &m, double &b,
                     std::vector<Point2D> &inliers) {
  if (fifo.size() < 2) {
    return;
  }
  std::random_device rd;
  // distrib0 must be one of the 5 most recent datapoints. i am not interested
  // in fit's of the older data

  auto gen = std::mt19937(rd());
  auto distrib0 = std::uniform_int_distribution<>(0, 5);
  auto distrib = std::uniform_int_distribution<>(0, ((data.size()) - (1)));
  auto best_inliers = std::vector<Point2D>();
  auto best_m = (0.);
  auto best_b = (0.);
  for (auto i = 0; i < RANSAC_MAX_ITERATIONS; i += 1) {
    auto idx1 = distrib(gen);
    auto idx2 = distrib0(gen);
    while (idx1 == idx2) {
      idx1 = distrib(gen);
    }
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
  inliers = best_inliers;
}
extern "C" {
#include "bme680.h"
#include "driver/uart.h"
#include "esp_system.h"
#include "hardware.h"
#include "pax_codecs.h"
#include "pax_gfx.h"
#include "soc/rtc_cntl_reg.h"
#include "sys/time.h"
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

#define CO2_UART UART_NUM_1
#define BUF_SIZE UART_FIFO_LEN

void uart_init() {
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

void measureBME() {
  {
    ESP_LOGE(TAG, "measure BME");
    auto temperature = (0.);
    auto humidity = (0.);
    auto pressure = (0.);
    auto bme = get_bme680();
    auto s = bme680_status_t();
    bme680_set_mode(get_bme680(), BME680_MEAS_FORCED);
    bme680_get_status(bme, &s);
    bme680_get_temperature(bme, &temperature);
    bme680_get_humidity(bme, &humidity, temperature);
    bme680_get_pressure(bme, &pressure, temperature);

    ESP_LOGE(
        TAG, "%s",
        fmt::format("  bme==nullptr='{}'  temperature='{}'  humidity='{}'  "
                    "pressure='{}'  s.new_data='{}'  s.gas_measuring='{}'  "
                    "s.measuring='{}'  s.gas_measuring_index='{}'  "
                    "s.gas_valid='{}'  s.heater_stable='{}'\n",
                    bme == nullptr, temperature, humidity, pressure, s.new_data,
                    s.gas_measuring, s.measuring, s.gas_measuring_index,
                    s.gas_valid, s.heater_stable)
            .c_str());
    if (((N_FIFO) - (1)) < fifo.size()) {
      fifoBME.pop_back();
    }
    auto tv_now = timeval();
    gettimeofday(&tv_now, nullptr);
    auto time_us = (tv_now.tv_sec + ((1.00e-6f) * tv_now.tv_usec));
    auto p = PointBME({.x = time_us,
                       .temperature = temperature,
                       .humidity = humidity,
                       .pressure = pressure});
    fifoBME.push_front(p);
  }
}

void measureCO2() {
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

void drawBME_temperature(pax_buf_t *buf) {
  auto time_ma = fifoBME[0].x;
  auto time_mi = fifoBME[((fifoBME.size()) - (1))].x;
  auto time_delta = ((time_ma) - (time_mi));
  auto scaleTime = [&](float x) -> float {
    auto res = ((318.f) * ((((x) - (time_mi))) / (time_delta)));
    if (res < (1.0f)) {
      res = (1.0f);
    }
    if ((318.f) < res) {
      res = (318.f);
    }
    return res;
  };
  auto min_max_y =
      std::minmax_element(fifoBME.begin(), fifoBME.end(),
                          [](const PointBME &p1, const PointBME &p2) {
                            return p1.temperature < p2.temperature;
                          });
  auto min_y = min_max_y.first->temperature;
  auto max_y = min_max_y.second->temperature;
  auto scaleHeight = [&](float v) -> float {
    auto mi = min_y;
    auto ma = max_y;
    auto res =
        ((61.f) + (59 * (((1.0f)) - (((((v) - (mi))) / (((ma) - (mi))))))));
    if (res < (61.f)) {
      res = (61.f);
    }
    if (119 < res) {
      res = 119;
    }
    return res;
  };
  // write latest measurement
  auto temperature = fifoBME[0].temperature;
  auto text_ = fmt::format("T={:2.2f}°C", ((1.0f) * temperature));
  auto font = pax_font_saira_condensed;
  auto text = text_.c_str();
  auto dims = pax_text_size(font, font->default_size, text);
  pax_draw_text(buf, 0xFFFFFFFF, font, font->default_size,
                ((((buf->width) - (dims.x))) / ((2.0f))),
                (-10 + ((0.50f) * ((61.f) + 119))), text);

  for (auto p : fifoBME) {
    // draw measurements as points

    for (auto i = 0; i < 3; i += 1) {
      for (auto j = 0; j < 3; j += 1) {
        pax_set_pixel(buf, pax_col_hsv(150, 180, 200),
                      (i + -1 + scaleTime(p.x)),
                      (j + -1 + scaleHeight(p.temperature)));
      }
    }
  }
}

void drawBME_humidity(pax_buf_t *buf) {
  auto time_ma = fifoBME[0].x;
  auto time_mi = fifoBME[((fifoBME.size()) - (1))].x;
  auto time_delta = ((time_ma) - (time_mi));
  auto scaleTime = [&](float x) -> float {
    auto res = ((318.f) * ((((x) - (time_mi))) / (time_delta)));
    if (res < (1.0f)) {
      res = (1.0f);
    }
    if ((318.f) < res) {
      res = (318.f);
    }
    return res;
  };
  auto min_max_y =
      std::minmax_element(fifoBME.begin(), fifoBME.end(),
                          [](const PointBME &p1, const PointBME &p2) {
                            return p1.humidity < p2.humidity;
                          });
  auto min_y = min_max_y.first->humidity;
  auto max_y = min_max_y.second->humidity;
  auto scaleHeight = [&](float v) -> float {
    auto mi = min_y;
    auto ma = max_y;
    auto res =
        ((121.f) + (59 * (((1.0f)) - (((((v) - (mi))) / (((ma) - (mi))))))));
    if (res < (121.f)) {
      res = (121.f);
    }
    if (179 < res) {
      res = 179;
    }
    return res;
  };
  // write latest measurement
  auto humidity = fifoBME[0].humidity;
  auto text_ = fmt::format("H={:2.1f}%", ((1.0f) * humidity));
  auto font = pax_font_saira_condensed;
  auto text = text_.c_str();
  auto dims = pax_text_size(font, font->default_size, text);
  pax_draw_text(buf, 0xFFFFFFFF, font, font->default_size,
                ((((buf->width) - (dims.x))) / ((2.0f))),
                (-10 + ((0.50f) * ((121.f) + 179))), text);

  for (auto p : fifoBME) {
    // draw measurements as points

    for (auto i = 0; i < 3; i += 1) {
      for (auto j = 0; j < 3; j += 1) {
        pax_set_pixel(buf, pax_col_hsv(80, 180, 200), (i + -1 + scaleTime(p.x)),
                      (j + -1 + scaleHeight(p.humidity)));
      }
    }
  }
}

void drawBME_pressure(pax_buf_t *buf) {
  auto time_ma = fifoBME[0].x;
  auto time_mi = fifoBME[((fifoBME.size()) - (1))].x;
  auto time_delta = ((time_ma) - (time_mi));
  auto scaleTime = [&](float x) -> float {
    auto res = ((318.f) * ((((x) - (time_mi))) / (time_delta)));
    if (res < (1.0f)) {
      res = (1.0f);
    }
    if ((318.f) < res) {
      res = (318.f);
    }
    return res;
  };
  auto min_max_y =
      std::minmax_element(fifoBME.begin(), fifoBME.end(),
                          [](const PointBME &p1, const PointBME &p2) {
                            return p1.pressure < p2.pressure;
                          });
  auto min_y = min_max_y.first->pressure;
  auto max_y = min_max_y.second->pressure;
  auto scaleHeight = [&](float v) -> float {
    auto mi = min_y;
    auto ma = max_y;
    auto res =
        ((181.f) + (59 * (((1.0f)) - (((((v) - (mi))) / (((ma) - (mi))))))));
    if (res < (181.f)) {
      res = (181.f);
    }
    if (239 < res) {
      res = 239;
    }
    return res;
  };
  // write latest measurement
  auto pressure = fifoBME[0].pressure;
  auto text_ = fmt::format("p={:4.2f}mbar", ((1.00e-2f) * pressure));
  auto font = pax_font_saira_condensed;
  auto text = text_.c_str();
  auto dims = pax_text_size(font, font->default_size, text);
  pax_draw_text(buf, 0xFFFFFFFF, font, font->default_size,
                ((((buf->width) - (dims.x))) / ((2.0f))),
                (-10 + ((0.50f) * ((181.f) + 239))), text);

  for (auto p : fifoBME) {
    // draw measurements as points

    for (auto i = 0; i < 3; i += 1) {
      for (auto j = 0; j < 3; j += 1) {
        pax_set_pixel(buf, pax_col_hsv(240, 180, 200),
                      (i + -1 + scaleTime(p.x)),
                      (j + -1 + scaleHeight(p.pressure)));
      }
    }
  }
}

void drawCO2(pax_buf_t *buf) {
  if (fifo.size() < 2) {
    return;
  }
  auto hue = 12;
  auto sat = 255;
  auto bright = 255;
  auto col = pax_col_hsv(hue, sat, bright);
  auto time_ma = fifo[0].x;
  auto time_mi = fifo[((fifo.size()) - (1))].x;
  auto time_delta = ((time_ma) - (time_mi));
  auto scaleTime = [&](float x) -> float {
    auto res = ((318.f) * ((((x) - (time_mi))) / (time_delta)));
    if (res < (1.0f)) {
      res = (1.0f);
    }
    if ((318.f) < res) {
      res = (318.f);
    }
    return res;
  };
  auto min_max_y = std::minmax_element(
      fifo.begin(), fifo.end(),
      [](const Point2D &p1, const Point2D &p2) { return p1.y < p2.y; });
  auto min_y = min_max_y.first->y;
  auto max_y = min_max_y.second->y;
  auto scaleHeight = [&](float v) -> float {
    // v is in the range 400 .. 5000
    // map to 0 .. 239

    auto mi = (4.00e+2f);
    auto ma = (max_y < (1.20e+3f)) ? ((1.20e+3f)) : ((5.00e+3f));
    auto res =
        ((1.0f) + (59 * (((1.0f)) - (((((v) - (mi))) / (((ma) - (mi))))))));
    if (res < (1.0f)) {
      res = (1.0f);
    }
    if (59 < res) {
      res = 59;
    }
    return res;
  };
  // write latest measurement
  auto co2 = fifo[0].y;
  auto text_ = fmt::format("CO2={:4.0f}ppm", co2);
  auto font = pax_font_saira_condensed;
  auto text = text_.c_str();
  auto dims = pax_text_size(font, font->default_size, text);
  pax_draw_text(buf, 0xFFFFFFFF, font, font->default_size,
                ((((buf->width) - (dims.x))) / ((2.0f))),
                (-10 + ((0.50f) * ((1.0f) + 59))), text);

  for (auto p : fifo) {
    // draw measurements as points

    for (auto i = 0; i < 3; i += 1) {
      for (auto j = 0; j < 3; j += 1) {
        pax_set_pixel(buf, pax_col_hsv(149, 180, 200),
                      (i + -1 + scaleTime(p.x)), (j + -1 + scaleHeight(p.y)));
      }
    }
  }
  {
    auto m = (0.);
    auto b = (0.);
    auto inliers = std::vector<Point2D>();
    auto hue = 128;
    auto sat = 255;
    auto bright = 200;
    auto col = pax_col_hsv(hue, sat, bright);
    ransac_line_fit(fifo, m, b, inliers);
    // draw the fit as line
    pax_draw_line(buf, col, scaleTime(time_mi),
                  scaleHeight((b + (m * time_mi))), scaleTime(time_ma),
                  scaleHeight((b + (m * time_ma))));
    // draw inliers as points
    for (auto p : inliers) {
      for (auto i = 0; i < 3; i += 1) {
        for (auto j = 0; j < 3; j += 1) {
          pax_set_pixel(buf, pax_col_hsv(0, 255, 255),
                        (i + -1 + scaleTime(p.x)), (j + -1 + scaleHeight(p.y)));
        }
      }
    }

    // compute when a value of 1200ppm is reached
    auto x0 = (((((1.20e+3)) - (b))) / (m));
    auto x0l = (((((5.00e+2)) - (b))) / (m));
    {
      auto text_ = fmt::format("m={:3.4f} b={:4.2f} xmi={:4.2f} xma={:4.2f}", m,
                               b, time_mi, time_ma);
      auto text = text_.c_str();
      auto font = pax_font_sky;
      auto dims = pax_text_size(font, font->default_size, text);
      pax_draw_text(buf, pax_col_hsv(160, 128, 128), font, font->default_size,
                    20, 80, text);

      {
        auto text_ = fmt::format("x0={:4.2f} x0l={:4.2f}", x0, x0l);
        auto text = text_.c_str();
        auto font = pax_font_sky;
        auto dims = pax_text_size(font, font->default_size, text);
        pax_draw_text(buf, pax_col_hsv(130, 128, 128), font, font->default_size,
                      20, 60, text);
      }
    }
    if (time_ma < x0) {
      // if predicted intersection time is in the future, print it
      auto time_value = static_cast<int>(((x0) - (time_ma)));
      auto hours = int(((time_value) / (3600)));
      auto minutes = int(((time_value % 3600) / (60)));
      auto seconds = time_value % 60;
      auto text_ = fmt::format("air room in (h:m:s) {:02d}:{:02d}:{:02d}",
                               hours, minutes, seconds);
      auto text = text_.c_str();
      auto font = pax_font_sky;
      auto dims = pax_text_size(font, font->default_size, text);
      pax_draw_text(buf, pax_col_hsv(30, 128, 128), font, font->default_size,
                    20, 140, text);

    } else {
      // if predicted intersection time is in the past, then predict when airing
      // should stop
      auto x0 = (((((5.00e+2)) - (b))) / (m));
      auto time_value = static_cast<int>(((x0) - (time_ma)));
      auto hours = int(((time_value) / (3600)));
      auto minutes = int(((time_value % 3600) / (60)));
      auto seconds = time_value % 60;
      auto text_ =
          fmt::format("air of room should stop in (h:m:s) {:02d}:{:02d}:{:02d}",
                      hours, minutes, seconds);
      auto text = text_.c_str();
      auto font = pax_font_sky;
      auto dims = pax_text_size(font, font->default_size, text);
      pax_draw_text(buf, pax_col_hsv(90, 128, 128), font, font->default_size,
                    20, 140, text);
    }
  }
}

void app_main() {
  ESP_LOGE(TAG, "welcome to the template app");
  auto ret = nvs_flash_init();
  if (((ESP_ERR_NVS_NO_FREE_PAGES == ret) ||
       (ESP_ERR_NVS_NEW_VERSION_FOUND == ret))) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);
  ESP_LOGE(TAG, "esp wifi mode sta");
  wifi_init_sta();

  bsp_init();
  bsp_rp2040_init();
  buttonQueue = get_rp2040()->queue;

  pax_buf_init(&buf, nullptr, 320, 240, PAX_BUF_16_565RGB);
  uart_init();
  bsp_bme680_init();
  bme680_set_mode(get_bme680(), BME680_MEAS_FORCED);
  bme680_set_oversampling(get_bme680(), BME680_OVERSAMPLING_X2,
                          BME680_OVERSAMPLING_X2, BME680_OVERSAMPLING_X2);
  while (1) {
    measureCO2();
    measureBME();
    auto hue = 129;
    auto sat = 0;
    auto bright = 0;
    auto col = pax_col_hsv(hue, sat, bright);
    pax_background(&buf, col);
    auto text_ = fmt::format("build 21:07:35 of Monday, 2023-04-03 (GMT+1)\n");
    auto text = text_.c_str();
    auto font = pax_font_sky;
    auto dims = pax_text_size(font, font->default_size, text);
    drawBME_temperature(&buf);
    drawBME_humidity(&buf);
    drawBME_pressure(&buf);
    pax_draw_text(&buf, 0xFFFFFFFF, font, font->default_size,
                  ((((buf.width) - (dims.x))) / ((2.0f))),
                  ((((buf.height) - (dims.y))) / ((2.0f))), text);
    {
      auto now = fifo[0].x;
      auto nowtext_ = fmt::format("now={:6.1f}", now);
      pax_draw_text(&buf, 0xFFFFFFFF, font, font->default_size, 20, 180,
                    nowtext_.c_str());
    }
    drawCO2(&buf);
    disp_flush();
    auto message = rp2040_input_message_t();
    xQueueReceive(buttonQueue, &message, 2);
    if (((RP2040_INPUT_BUTTON_HOME == message.input) && (message.state))) {
      exit_to_launcher();
    }
  }
}
};
