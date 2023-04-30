// no preamble
#define FMT_HEADER_ONLY
#include "OfdmReceiver.h"
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
OfdmReceiver::OfdmReceiver() {}
std::vector<Cplx> OfdmReceiver::receive(const std::vector<Cplx> &receivedData) {
  auto start_index = schmidlCoxSynchronization(receivedData);
  auto in = std::array<Cplx, FFT_SIZE>();
  auto out = std::array<Cplx, FFT_SIZE>();
  auto *fftPlan =
      fftw_plan_dft_1d(FFT_SIZE, reinterpret_cast<fftw_complex *>(in.data()),
                       reinterpret_cast<fftw_complex *>(out.data()),
                       FFTW_FORWARD, FFTW_ESTIMATE);
  auto fftData = std::vector<Cplx>(((FFT_SIZE) * (SYMBOLS)));
  for (auto symbol = 0; (symbol) < (SYMBOLS); (symbol) += (1)) {
    std::copy(((receivedData.begin()) + (start_index) +
               (((symbol) * (((FFT_SIZE) + (CP_SIZE))))) + (CP_SIZE)),
              ((receivedData.begin()) + (start_index) +
               (((((1) + (symbol))) * (((FFT_SIZE) + (CP_SIZE)))))),
              in.data());
    fftw_execute(fftPlan);
    std::copy(out.data(), ((out.data()) + (FFT_SIZE)),
              ((fftData.begin()) + (((symbol) * (FFT_SIZE)))));
  }
  fftw_destroy_plan(fftPlan);
  return fftData;
}
size_t
OfdmReceiver::schmidlCoxSynchronization(const std::vector<Cplx> &receivedData) {
  auto R = std::vector<double>(((FFT_SIZE) + (CP_SIZE)), 0.0);
  auto M = std::vector<double>(((FFT_SIZE) + (CP_SIZE)), 0.0);
  auto P = double(0.0);
  for (auto i = 0; (i) < (((FFT_SIZE) + (CP_SIZE))); (i) += (1)) {
    R[i] = std::abs(((receivedData[((i) + (FFT_SIZE) + (CP_SIZE))]) *
                     (std::conj(receivedData[i]))));

    M[i] = ((std::norm(receivedData[((i) + (FFT_SIZE) + (CP_SIZE))])) +
            (std::norm(receivedData[i])));

    (P) += (M[i]);
  }
  auto max_metric = double(-1.0);
  auto start_index = size_t(0);
  for (auto i = 0; (i) < (((FFT_SIZE) + (CP_SIZE))); (i) += (1)) {
    auto metric = ((R[i]) / (((M[i]) / (P))));
    if (((max_metric) < (metric))) {
      max_metric = metric;
      start_index = i;
    }
  }
  return start_index;
}
