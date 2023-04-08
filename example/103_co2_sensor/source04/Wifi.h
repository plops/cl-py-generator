#ifndef WIFI_H
#define WIFI_H

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"

class Wifi  {
        private:
        enum { WIFI_CONNECTED_BIT=BIT0, WIFI_FAIL_BIT=BIT1, EXAMPLE_ESP_MAXIMUM_RETRY=7 };
        int s_retry_num;
        EventGroupHandle_t s_wifi_event_group;
        void event_handler (void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data)       ;  
        public:
        explicit  Wifi ()       ;  
};

#endif /* !WIFI_H */