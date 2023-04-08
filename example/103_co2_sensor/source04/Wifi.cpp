// no preamble
extern "C" {
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"
#include "secret.h"
#include <arpa/inet.h>
// event group should allow two different events
// 1) we are connected to access point with an ip
// 2) we failed to connect after a maximum amount of retries
};
#define FMT_HEADER_ONLY
#include "core.h"

#include "Wifi.h"
void Wifi::event_handler(void *arg, esp_event_base_t event_base,
                         int32_t event_id, void *event_data) {
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
Wifi::Wifi() : s_retry_num(0) {
  s_wifi_event_group = xEventGroupCreate();

  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_sta();
  auto cfg = wifi_init_config_t WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&cfg));
  auto instance_any_id = esp_event_handler_instance_t();
  auto instance_got_ip = esp_event_handler_instance_t();
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      WIFI_EVENT, ESP_EVENT_ANY_ID,
      reinterpret_cast<esp_event_handler_t>(&Wifi::event_handler), nullptr,
      &instance_any_id));
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      IP_EVENT, IP_EVENT_STA_GOT_IP,
      reinterpret_cast<esp_event_handler_t>(&Wifi::event_handler), nullptr,
      &instance_got_ip));
  wifi_config_t wifi_config = {};
  const char *ssid_str = "mi";
  std::memcpy(wifi_config.sta.ssid, ssid_str, std::strlen(ssid_str));
  const char *password_str = WIFI_SECRET;
  std::memcpy(wifi_config.sta.password, password_str,
              std::strlen(password_str));
  wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

  wifi_config.sta.pmf_cfg.capable = true;

  wifi_config.sta.pmf_cfg.required = false;

  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
  ESP_ERROR_CHECK(esp_wifi_start());
  fmt::print("wait until connection is established or connection failed "
             "s_retry_num times  s_retry_num='{}'\n",
             s_retry_num);
  auto bits = xEventGroupWaitBits(s_wifi_event_group,
                                  ((WIFI_CONNECTED_BIT) | (WIFI_FAIL_BIT)),
                                  pdFALSE, pdFALSE, portMAX_DELAY);
  if (((WIFI_CONNECTED_BIT) & (bits))) {
    fmt::print("connected to ap\n");
  } else {
    if (((WIFI_FAIL_BIT) & (bits))) {
      fmt::print("connection to ap failed\n");
    } else {
      fmt::print("unexpected event\n");
    }
  }
  ESP_ERROR_CHECK(esp_event_handler_instance_unregister(
      IP_EVENT, IP_EVENT_STA_GOT_IP, instance_got_ip));
  ESP_ERROR_CHECK(esp_event_handler_instance_unregister(
      WIFI_EVENT, ESP_EVENT_ANY_ID, instance_any_id));
  vEventGroupDelete(s_wifi_event_group);
}
