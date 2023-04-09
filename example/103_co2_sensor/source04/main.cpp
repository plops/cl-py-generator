#include "DataTypes.h"
#include "Display.h"
#include "Ransac.h"
#include "TcpConnection.h"
#include "Uart.h"
#include "Wifi.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <random>
#include <vector>
#define FMT_HEADER_ONLY
#include "core.h"
std::deque<Point2D> fifo;
std::deque<PointBME> fifoBME;
extern "C" {
#include "bme680.h"
#include "driver/uart.h"
#include "esp_system.h"
#include "hardware.h"
#include "nvs_flash.h"
#include "pax_codecs.h"
#include "pax_gfx.h"
#include "soc/rtc_cntl_reg.h"
#include "sys/time.h"
#include <esp_log.h>

static const char *TAG = "mch2022-co2-app";
xQueueHandle buttonQueue;

void exit_to_launcher() {
  REG_WRITE(RTC_CNTL_STORE0_REG, 0);
  esp_restart();
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

void drawBME_temperature(Display &display) {
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
  display.small_text(text_, -1, (-10 + ((0.50f) * ((61.f) + 119))));

  for (auto p : fifoBME) {
    // draw measurements as points

    for (auto i = 0; i < 3; i += 1) {
      for (auto j = 0; j < 3; j += 1) {
        display.set_pixel((i + -1 + scaleTime(p.x)),
                          (j + -1 + scaleHeight(p.temperature)), 150, 180, 200);
      }
    }
  }
}

void drawBME_humidity(Display &display) {
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
  display.small_text(text_, -1, (-10 + ((0.50f) * ((121.f) + 179))));

  for (auto p : fifoBME) {
    // draw measurements as points

    for (auto i = 0; i < 3; i += 1) {
      for (auto j = 0; j < 3; j += 1) {
        display.set_pixel((i + -1 + scaleTime(p.x)),
                          (j + -1 + scaleHeight(p.humidity)), 80, 180, 200);
      }
    }
  }
}

void drawBME_pressure(Display &display) {
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
  display.small_text(text_, -1, (-10 + ((0.50f) * ((181.f) + 239))));

  for (auto p : fifoBME) {
    // draw measurements as points

    for (auto i = 0; i < 3; i += 1) {
      for (auto j = 0; j < 3; j += 1) {
        display.set_pixel((i + -1 + scaleTime(p.x)),
                          (j + -1 + scaleHeight(p.pressure)), 240, 180, 200);
      }
    }
  }
}

void drawCO2(Display &display) {
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
  display.large_text(text_, -1, (-10 + ((0.50f) * ((1.0f) + 59))));

  for (auto p : fifo) {
    // draw measurements as points

    for (auto i = 0; i < 3; i += 1) {
      for (auto j = 0; j < 3; j += 1) {
        display.set_pixel((i + -1 + scaleTime(p.x)),
                          (j + -1 + scaleHeight(p.y)), 149, 180, 200);
      }
    }
  }
  {
    auto ransac = Ransac(fifo);
    auto m = ransac.GetM();
    auto b = ransac.GetB();
    auto inliers = ransac.GetInliers();
    auto hue = 128;
    auto sat = 255;
    auto bright = 200;
    auto col = pax_col_hsv(hue, sat, bright);
    // draw the fit as line
    display.line(scaleTime(time_mi), scaleHeight((b + (m * time_mi))),
                 scaleTime(time_ma), scaleHeight((b + (m * time_ma))), 188, 255,
                 200);
    // draw inliers as points
    for (auto p : inliers) {
      for (auto i = 0; i < 3; i += 1) {
        for (auto j = 0; j < 3; j += 1) {
          display.set_pixel((i + -1 + scaleTime(p.x)),
                            (j + -1 + scaleHeight(p.y)), 0, 255, 255);
        }
      }
    }

    // compute when a value of 1200ppm is reached
    auto x0 = (((((1.20e+3)) - (b))) / (m));
    auto x0l = (((((5.00e+2)) - (b))) / (m));
    {
      display.small_text(
          fmt::format("m={:3.4f} b={:4.2f} xmi={:4.2f} xma={:4.2f}", m, b,
                      time_mi, time_ma),
          20, 80, 160, 128, 128);
      {
        display.small_text(fmt::format("x0={:4.2f} x0l={:4.2f}", x0, x0l), 20,
                           60, 130, 128, 128);
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
      display.small_text(text_, 20, 140, 30, 128, 128);

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
      display.small_text(text_, 20, 140, 90, 128, 128);
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
  Wifi wifi;
  TcpConnection tcp;

  bsp_init();
  bsp_rp2040_init();
  buttonQueue = get_rp2040()->queue;

  Display display;
  Uart uart;
  bsp_bme680_init();
  bme680_set_mode(get_bme680(), BME680_MEAS_FORCED);
  bme680_set_oversampling(get_bme680(), BME680_OVERSAMPLING_X2,
                          BME680_OVERSAMPLING_X2, BME680_OVERSAMPLING_X2);
  while (1) {
    uart.measureCO2(fifo);
    measureBME();
    display.background(129, 0, 0);
    drawBME_temperature(display);
    drawBME_humidity(display);
    drawBME_pressure(display);
    display.small_text(
        fmt::format("build 10:40:41 of Sunday, 2023-04-09 (GMT+1)\n"));
    {
      auto now = fifo[0].x;
      display.small_text(fmt::format("now={:6.1f}", now), 20, 180);
    }
    drawCO2(display);
    display.flush();
    auto message = rp2040_input_message_t();
    xQueueReceive(buttonQueue, &message, 2);
    if (((RP2040_INPUT_BUTTON_HOME == message.input) && (message.state))) {
      exit_to_launcher();
    }
  }
}
};
