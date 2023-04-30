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
  auto *fftPlan =
      fftw_plan_dft_1d(FFT_SIZE, reinterpret_cast<fftw_complex *>(in.data()),
                       reinterpret_cast<fftw_complex *>(out.data()),
                       FFTW_FORWARD, FFTW_ESTIMATE);
  auto ifftData = std::vector<Cplx>(((FFT_SIZE) * (SYMBOLS)));
  for (auto symbol = 0; (symbol) < (SYMBOLS); (symbol) += (1)) {
    for (auto i = 0; (i) < (FFT_SIZE); (i) += (1)) {
      in[i] = data[((i) + (((symbol) * (FFT_SIZE))))];
    }
    fftw_execute(fftPlan);
    std::copy(out.data(), ((out.data()) + (FFT_SIZE)),
              ((ifftData.begin()) + (((symbol) * (FFT_SIZE)))));
  }
  // insert cyclic prefix and preamble
  auto transmittedData =
      std::vector<Cplx>(((((SYMBOLS) * (((FFT_SIZE) + (CP_SIZE))))) +
                         (((2) * (((FFT_SIZE) + (CP_SIZE)))))));
  std::copy(preamble.begin(), preamble.end(), transmittedData.begin());
  for (auto symbol = 0; (symbol) < (SYMBOLS); (symbol) += (1)) {
    std::copy(((ifftData.begin()) + (((symbol) * (FFT_SIZE)))),
              ((ifftData.begin()) + (((symbol) * (FFT_SIZE))) + (CP_SIZE)),
              ((transmittedData.begin()) +
               (((2) * (((FFT_SIZE) + (CP_SIZE))))) +
               (((symbol) * (((FFT_SIZE) + (CP_SIZE)))))));
    std::copy(((ifftData.begin()) + (((symbol) * (FFT_SIZE))) + (CP_SIZE)),
              ((ifftData.begin()) + (((((1) + (symbol))) * (FFT_SIZE)))),
              ((transmittedData.begin()) +
               (((2) * (((FFT_SIZE) + (CP_SIZE))))) +
               (((symbol) * (((FFT_SIZE) + (CP_SIZE))))) + (CP_SIZE)));
  }
  fftw_destroy_plan(fftPlan);
  return transmittedData;
}
std::vector<Cplx> OfdmTransmitter::generatePreamble() {
  auto preamble = std::vector<Cplx>(((2) * (((FFT_SIZE) + (CP_SIZE)))));
  auto random_symbols = std::vector<Cplx>(((FFT_SIZE) / (2)));
  auto generator = std::default_random_engine();
  auto distribution = std::uniform_real_distribution<double>(-1, 1);
  for (auto i = 0; (i) < (((FFT_SIZE) / (2))); (i) += (1)) {
    random_symbols[i] = Cplx(distribution(generator), distribution(generator));
  }
  auto in = std::array<Cplx, FFT_SIZE>();
  auto out = std::array<Cplx, FFT_SIZE>();
  auto *ifftPlan =
      fftw_plan_dft_1d(FFT_SIZE, reinterpret_cast<fftw_complex *>(in.data()),
                       reinterpret_cast<fftw_complex *>(out.data()),
                       FFTW_BACKWARD, FFTW_ESTIMATE);
  for (auto i = 0; (i) < (((FFT_SIZE) / (2))); (i) += (1)) {
    in[i] = random_symbols[i];
    in[((i) + (((FFT_SIZE) / (2))))] = std::conj(random_symbols[i]);
  }
  fftw_execute(ifftPlan);
  // Add cyclic prefix and copy repeated preambels
  std::copy(((out.data()) + (FFT_SIZE) + (-CP_SIZE)),
            ((out.data()) + (FFT_SIZE)), preamble.begin());
  std::copy(out.data(), ((out.data()) + (FFT_SIZE)),
            ((preamble.begin()) + (CP_SIZE)));
  std::copy(preamble.begin(), ((preamble.begin()) + (FFT_SIZE) + (CP_SIZE)),
            ((preamble.begin()) + (FFT_SIZE) + (CP_SIZE)));
  fftw_destroy_plan(ifftPlan);
  return preamble;
}
