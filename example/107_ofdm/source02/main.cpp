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
}
