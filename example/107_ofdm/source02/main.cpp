#include "OfdmReceiver.h"
#include "OfdmTransmitter.h"
#include <random>
#include <vector>
#define FMT_HEADER_ONLY
#include "core.h"

int main(int argc, char **argv) {
  auto data = std::vector<Cplx>(((FFT_SIZE) * (SYMBOLS)));
  auto generator = std::default_random_engine();
  auto distribution = std::uniform_real_distribution<double>(-1.0, 1.0);
  for (auto i = 0; (i) < (((FFT_SIZE) * (SYMBOLS))); (i) += (1)) {
    data[i] = Cplx(distribution(generator), distribution(generator));
  }
  auto transmitter = OfdmTransmitter();
  auto transmittedData = transmitter.transmit(data);
  auto receiver = OfdmReceiver();
  auto receivedData = receiver.receive(transmittedData);
  auto err = double(0.0);
  for (auto i = 0; (i) < (((SYMBOLS) * (FFT_SIZE))); (i) += (1)) {
    (err) += (std::norm(((data[i]) - (receivedData[i]))));
  }
  err = std::sqrt(((err) / (((FFT_SIZE) * (SYMBOLS)))));

  fmt::print("  err='{}'\n", err);
  return 0;
}
