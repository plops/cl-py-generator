#ifndef WIFI_H
#define WIFI_H

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

class Wifi  {
        private:
        enum { WIFI_CONNECTED_BIT=BIT0, WIFI_FAIL_BIT=BIT1, EXAMPLE_ESP_MAXIMUM_RETRY=7 };
        static int s_retry_num;
        static EventGroupHandle_t s_wifi_event_group;
        static void event_handler (void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data)       ;  
        public:
        explicit  Wifi ()       ;  
};

#endif /* !WIFI_H */