// no preamble
#define FMT_HEADER_ONLY
#include "OfdmTransmitter.h"
#include "core.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <iostream>
#include <random>
#include <vector>
OfdmTransmitter::OfdmTransmitter() {}
std::vector<Cplx> OfdmTransmitter::transmit(const std::vector<Cplx> &data) {
  auto preamble = generatePreamble();
  auto in = std::array<Cplx, FFT_SIZE>();
  auto out = std::array<Cplx, FFT_SIZE>();
  auto fftPlan =
      fftw_plan_dft_1d(FFT_SIZE, reinterpret_cast<fftw_complex *>(in.data()),
                       reinterpret_cast<fftw_complex *>(out.data()),
                       FFTW_FORWARD, FFTW_ESTIMATE);
  auto ifftData = std::vector<Cplx>(((FFT_SIZE) * (SYMBOLS)));
  for (auto symbol = 0; (symbol) < (SYMBOLS); (symbol) += (1)) {
    for (auto i = 0; (i) < (FFT_SIZE); (i) += (1)) {
      in[i] = data[((i) + (((symbol) * (FFT_SIZE))))];
    }
    fftw_execute(fftPlan);
    std::copy(out, ((out) + (FFT_SIZE)),
              ((ifftData.begin()) + (((symbol) * (FFT_SIZE)))));
  }
}
std::vector<Cplx> OfdmTransmitter::generatePreamble() {}
