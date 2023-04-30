// no preamble
extern "C" {
#include "data.pb.h"
#include "esp_netif.h"
#include "esp_netif_types.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <pb_decode.h>
#include <pb_encode.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
};
#include <array>
#define FMT_HEADER_ONLY
#include "core.h"

#include "TcpConnection.h"
bool TcpConnection::read_callback(pb_istream_t *stream, uint8_t *buf,
                                  size_t count) {
  auto fd = reinterpret_cast<intptr_t>(stream->state);

  if (((0) == (count))) {
    return true;
  }
  // operation should block until full request is satisfied. may still return
  // less than requested (upon signal, error or disconnect)

  auto result = recv(fd, buf, count, MSG_WAITALL);
  if (((result) < (0))) {
    fmt::print("recv failed  strerror(errno)='{}'\n", strerror(errno));
  }
  fmt::print("read_callback  count='{}'  result='{}'\n", count, result);
  for (auto i = 0; (i) < (count); (i) += (1)) {
    fmt::print("{:02x} ", buf[i]);
  }
  fmt::print("\n");
  if (((0) == (result))) {
    // EOF
    stream->bytes_left = 0;
  }
  return (count) == (result);
}
bool TcpConnection::write_callback(pb_ostream_t *stream, const pb_byte_t *buf,
                                   size_t count) {
  auto fd = reinterpret_cast<intptr_t>(stream->state);

  auto res = send(fd, buf, count, 0);
  if (((res) < (0))) {
    fmt::print("send failed  strerror(errno)='{}'\n", strerror(errno));
  }
  return (count) == (res);
}
void TcpConnection::set_socket_timeout(int fd, float timeout_seconds) {
  auto timeout = timeval();
  timeout.tv_sec = static_cast<int>(timeout_seconds);
  timeout.tv_usec = static_cast<int>(
      ((1000000) *
       (((timeout_seconds) - (static_cast<float>(timeout.tv_sec))))));

  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}
pb_istream_t TcpConnection::pb_istream_from_socket(int fd) {
  // note: the designated initializer syntax requires C++20

  auto stream = pb_istream_t();
  stream.callback = &TcpConnection::read_callback;

  stream.state = reinterpret_cast<void *>(static_cast<intptr_t>(fd));

  stream.bytes_left = SIZE_MAX;

  return stream;
}
pb_ostream_t TcpConnection::pb_ostream_from_socket(int fd) {
  auto stream = pb_ostream_t();
  stream.callback = &TcpConnection::write_callback;

  stream.state = reinterpret_cast<void *>(static_cast<intptr_t>(fd));

  stream.max_size = SIZE_MAX;

  stream.bytes_written = 0;

  return stream;
}
void TcpConnection::send_data(float pressure, float humidity, float temperature,
                              float co2_concentration) {
  auto s = socket(AF_INET, SOCK_STREAM, 0);
  auto port = u16_t(12345);
  auto server_addr =
      sockaddr_in({.sin_family = AF_INET, .sin_port = htons(port)});
  if (((s) < (0))) {
    fmt::print("error creating socket  strerror(errno)='{}'\n",
               strerror(errno));
    return;
  }
  set_socket_timeout(s, (2.0f));
  // i use the esp32 in a wifi network that is provided by a phone. the ip
  // address of the clients can sometimes change. it seems that the server keeps
  // the last part (122), though. so in order to get the server ip i will first
  // look at the esp32 ip and replace the last number with 122.
  auto *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
  auto ip_info = esp_netif_ip_info_t();
  esp_netif_get_ip_info(netif, &ip_info);
  auto client_ip = std::array<char, INET_ADDRSTRLEN>();
  inet_ntop(AF_INET, &ip_info.ip, client_ip.data(), INET_ADDRSTRLEN);
  auto client_ip_str = std::string(client_ip.data());
  auto server_ip_base =
      client_ip_str.substr(0, ((client_ip_str.rfind('.')) + (1)));
  auto server_ip = ((server_ip_base) + ("122"));
  inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr);
  fmt::print("connect to  client_ip_str='{}'  server_ip='{}'  port='{}'\n",
             client_ip_str, server_ip, port);

  if (((connect(s, reinterpret_cast<sockaddr *>(&server_addr),
                sizeof(server_addr))) < (0))) {
    fmt::print("error connecting  strerror(errno)='{}'\n", strerror(errno));
    close(s);
    return;
  }
  fmt::print("send measurement values in a DataResponse message\n");
  auto now = time_t();
  static uint64_t count = 0;
  time(&now);
  auto omsg = DataResponse({.index = count,
                            .datetime = now,
                            .pressure = pressure,
                            .humidity = humidity,
                            .temperature = temperature,
                            .co2_concentration = co2_concentration});
  auto output = pb_ostream_from_socket(s);
  (count)++;
  if ((!(pb_encode(&output, DataResponse_fields, &omsg)))) {
    fmt::print("error encoding\n");
  }

  fmt::print("close the output stream of the socket, so that the server "
             "receives a FIN packet\n");
  shutdown(s, SHUT_WR);
  fmt::print("read DataRequest\n");
  auto imsg = DataRequest({});
  auto input = pb_istream_from_socket(s);
  if ((!(pb_decode(&input, DataRequest_fields, &imsg)))) {
    fmt::print("error decoding\n");
  }
  fmt::print("  imsg.count='{}'  imsg.start_index='{}'\n", imsg.count,
             imsg.start_index);
  close(s);
}
TcpConnection::TcpConnection() {}
