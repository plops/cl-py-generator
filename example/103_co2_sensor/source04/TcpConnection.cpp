// no preamble
extern "C" {
#include "data.pb.h"
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

  return (count) == (send(fd, buf, count, 0));
}
void TcpConnection::set_socket_timeout(int fd, float timeout_seconds) {
  auto timeout = timeval();
  timeout.tv_sec = static_cast<int>(timeout_seconds);
  timeout.tv_usec =
      stati_cast<int>(((1000000) * (((timeout_seconds) - (timeout.tv_sec)))));

  setsockopt(fd, SOL_SOCKET, SO_RCVTIME0, &timeout, sizeof(timeout));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIME0, &timeout, sizeof(timeout));
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
  auto server_addr =
      sockaddr_in({.sin_family = AF_INET, .sin_port = htons(12345)});
  set_socket_timeout(s, (2.0f));
  inet_pton(AF_INET, "192.168.2.122", &server_addr.sin_addr);
  if ((connect(s, reinterpret_cast<sockaddr *>(&server_addr),
               sizeof(server_addr)))) {
    fmt::print("error connecting\n");
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
}
TcpConnection::TcpConnection() {}
