// no preamble
#define FMT_HEADER_ONLY
#include "core.h"
extern "C" {
#include <bme680.h>
#include <esp_log.h>
#include <hardware.h>
};

#include "BmeSensor.h"
void BmeSensor::measureBME(std::deque<PointBME> &fifoBME,
                           std::deque<Point2D> &fifo) {
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
        fmt::format("  (bme)==(nullptr)='{}'  temperature='{}'  humidity='{}'  "
                    "pressure='{}'  s.new_data='{}'  s.gas_measuring='{}'  "
                    "s.measuring='{}'  s.gas_measuring_index='{}'  "
                    "s.gas_valid='{}'  s.heater_stable='{}'\n",
                    (bme) == (nullptr), temperature, humidity, pressure,
                    s.new_data, s.gas_measuring, s.measuring,
                    s.gas_measuring_index, s.gas_valid, s.heater_stable)
            .c_str());
    if (((((N_FIFO) - (1))) < (fifo.size()))) {
      fifoBME.pop_back();
    }
    auto tv_now = timeval();
    gettimeofday(&tv_now, nullptr);
    auto time_us = ((tv_now.tv_sec) + ((((1.00e-6f)) * (tv_now.tv_usec))));
    auto p = PointBME({.x = time_us,
                       .temperature = temperature,
                       .humidity = humidity,
                       .pressure = pressure});
    fifoBME.push_front(p);
  }
}
BmeSensor::BmeSensor() {
  bsp_bme680_init();
  bme680_set_mode(get_bme680(), BME680_MEAS_FORCED);
  bme680_set_oversampling(get_bme680(), BME680_OVERSAMPLING_X2,
                          BME680_OVERSAMPLING_X2, BME680_OVERSAMPLING_X2);
}
