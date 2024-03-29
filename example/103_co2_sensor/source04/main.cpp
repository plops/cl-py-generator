#include "BmeSensor.h"
#include "DataTypes.h"
#include "Display.h"
#include "Graph.h"
#include "TcpConnection.h"
#include "Uart.h"
#include "Wifi.h"
#include <deque>
#define FMT_HEADER_ONLY
#include "core.h"
std::deque<Point2D> fifo;
std::deque<PointBME> fifoBME;
extern "C" {
#include "hardware.h"
#include "nvs_flash.h"
#include "soc/rtc_cntl_reg.h"
#include <esp_log.h>

static const char *TAG = "mch2022-co2-app";
xQueueHandle buttonQueue;

void exit_to_launcher() {
  REG_WRITE(RTC_CNTL_STORE0_REG, 0);
  esp_restart();
}

void app_main() {
  ESP_LOGE(TAG, "welcome to the template app");
  auto ret = nvs_flash_init();
  if (((((ESP_ERR_NVS_NO_FREE_PAGES) == (ret)) ||
        ((ESP_ERR_NVS_NEW_VERSION_FOUND) == (ret))))) {
    fmt::print("perform nvs_flash_erase\n");
    ESP_ERROR_CHECK(nvs_flash_erase());
    fmt::print("perform nvs_flash_init\n");
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
  Graph graph(display, fifo, fifoBME);
  Uart uart;
  BmeSensor bme;
  while (1) {
    uart.measureCO2(fifo);
    bme.measureBME(fifoBME, fifo);
    auto co2_val = fifo.front();
    auto bme_val = fifoBME.front();
    tcp.send_data(bme_val.pressure, bme_val.humidity, bme_val.temperature,
                  co2_val.y);

    display.background(129, 0, 0);
    graph.temperature();
    graph.humidity();
    graph.pressure();
    display.small_text(
        fmt::format("build 00:38:13 of Monday, 2023-05-01 (GMT+1)\n"));
    {
      auto now = fifo[0].x;
      display.small_text(fmt::format("now={:6.1f}", now), 20, 180);
    }
    graph.carbon_dioxide();
    display.flush();
    auto message = rp2040_input_message_t();
    xQueueReceive(buttonQueue, &message, 2);
    if (((((RP2040_INPUT_BUTTON_HOME) == (message.input)) &&
          (message.state)))) {
      exit_to_launcher();
    }
  }
}
};
