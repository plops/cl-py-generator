#define FMT_HEADER_ONLY
#include "core.h"
#include <cmath>
#include <deque>
#include <random>
#include <vector>
const int N_FIFO = 12;
const int RANSAC_MAX_ITERATIONS = 12;
const float RANSAC_INLIER_THRESHOLD = 0.1;
const int RANSAC_MIN_INLIERS = 50;
struct Point2D {
  double x;
  double y;
};
typedef struct Point2D Point2D;

std::deque<Point2D> fifo(N_FIFO, {0.0, 0.0});

double distance(Point2D p, double m, double b) {
  // division normalizes distance, so that it is independent of the slope of the
  // line
  return ((abs(((p.y) - (((m * p.x) + b))))) / (sqrt((1 + (m * m)))));
}

void ransac_line_fit(std::deque<Point2D> &data, double &m, double &b) {
  if (fifo.size() < 2) {
    return;
  }
  std::random_device rd;
  auto gen = std::mt19937(rd());
  auto distrib = std::uniform_int_distribution<>(0, ((data.size()) - (1)));
  auto best_inliers = std::vector<Point2D>();
  auto best_m = (0.);
  auto best_b = (0.);
  for (auto i = 0; i < RANSAC_MAX_ITERATIONS; i += 1) {
    // line model needs two points, so randomly select two points and compute
    // model parameters

    auto idx1 = distrib(gen);
    auto idx2 = distrib(gen);
    while (idx1 == idx2) {
      idx1 = distrib(gen);
    }
    auto p1 = data[idx1];
    auto p2 = data[idx2];
    auto m = ((((p2.y) - (p1.y))) / (((p2.x) - (p1.x))));
    auto b = ((p1.y) - ((m * p1.x)));
    auto inliers = std::vector<Point2D>();
    for (auto &p : data) {
      if (distance(p, m, b) < RANSAC_INLIER_THRESHOLD) {
        inliers.push_back(p);
      }
    };
    fmt::print("  idx1='{}'  idx2='{}'  data.size()='{}'  inliers.size()='{}'  "
               "m='{}'  b='{}'\n",
               idx1, idx2, data.size(), inliers.size(), m, b);
    if (RANSAC_MIN_INLIERS < inliers.size()) {
      auto sum_x = (0.);
      auto sum_y = (0.);
      for (auto &p : inliers) {
        sum_x += p.x;
        sum_y += p.y;
      };
      auto avg_x = ((sum_x) / (inliers.size()));
      auto avg_y = ((sum_y) / (inliers.size()));
      auto var_x = (0.);
      auto cov_xy = (0.);
      for (auto &p : inliers) {
        var_x += (((p.x) - (avg_x)) * ((p.x) - (avg_x)));
        cov_xy += (((p.x) - (avg_x)) * ((p.y) - (avg_y)));
      };
      auto m = ((cov_xy) / (var_x));
      auto b = ((avg_y) - ((m * avg_x)));
      if (best_inliers.size() < inliers.size()) {
        best_inliers = inliers;
        best_m = m;
        best_b = b;
      }
    }
  }
  m = best_m;
  b = best_b;
}

int main(int argc, char **argv) {
  auto m0 = (0.100000000000000000000000000000);
  auto b0 = (23.);
  fmt::print("  m0='{}'  b0='{}'\n", m0, b0);
  for (auto i = 0; i < N_FIFO; i += 1) {
    auto x = (1.0 * i);
    auto y = (b0 + (m0 * x));
    auto p = Point2D({.x = x, .y = y});
    if (((N_FIFO) - (1)) < fifo.size()) {
      fifo.pop_back();
    }
    fifo.push_front(p);
  }

  for (auto i = 0; i < fifo.size(); i += 1) {
    fmt::print("  i='{}'  fifo[i].x='{}'  fifo[i].y='{}'\n", i, fifo[i].x,
               fifo[i].y);
  }
  auto m = (0.);
  auto b = (0.);
  ransac_line_fit(fifo, m, b);
  fmt::print("  m='{}'  b='{}'\n", m, b);
}
