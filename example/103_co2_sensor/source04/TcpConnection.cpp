// no preamble
extern "C" {
#include "esp_netif_types.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"
#include <arpa/inet.h>
};
#define FMT_HEADER_ONLY
#include "core.h"

#include "TcpConnection.h"
TcpConnection::TcpConnection() {
  auto port = 12345;
  auto ip_address = "192.168.120.122";
  auto addr = ([port, ip_address]() -> sockaddr_in {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    inet_pton(AF_INET, ip_address, &addr.sin_addr);
    return addr;
  })();
  auto domain = AF_INET;
  auto type = SOCK_STREAM;
  auto protocol = 0;
  auto sock = socket(domain, type, protocol);
  if (sock < 0) {
    fmt::print("failed to create socket\n");
  }
  if ((0) != (connect(sock, reinterpret_cast<const sockaddr *>(&addr),
                      sizeof(addr)))) {
    fmt::print("failed to connect to socket\n");
  }
  fmt::print("connected to tcp server\n");
  constexpr auto buffer_size = 1024;
  auto read_buffer = std::array<char, buffer_size>{};
  auto r = read(sock, read_buffer.data(), ((read_buffer.size()) - (1)));
  if (r < 0) {
    fmt::print("failed to read data from socket\n");
  }
  read_buffer[r] = '\0';

  fmt::print("received data from server  read_buffer.data()='{}'\n",
             read_buffer.data());
}
