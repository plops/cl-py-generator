#define FMT_HEADER_ONLY
#include "core.h"
#include "matplotlibcpp.h"
#include <chrono>
#include <cmath>
#include <deque>
#include <random>
#include <vector>
namespace plt = matplotlibcpp;
const int N_FIFO = 240;
const int RANSAC_MAX_ITERATIONS = 240;
const float RANSAC_INLIER_THRESHOLD = 0.1;
const int RANSAC_MIN_INLIERS = 24;
struct Point2D {
  double x;
  double y;
};
typedef struct Point2D Point2D;

std::deque<Point2D> fifo(N_FIFO, {0.0, 0.0});
class Line {
public:
  Line(double m, double b) : m_(m), b_(b) {}
  Point2D point(double x) { return {x, ((m_ * x) + b_)}; }

private:
  double m_;
  double b_;
};

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
  auto m0 = (1.0);
  auto b0 = (2.0);
  auto noise_stddev = (0.100000000000000000000000000000);
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(0.0, noise_stddev);
  for (auto i = 0; i < N_FIFO; i += 1) {
    auto x = (((1.0 * i)) / (N_FIFO));
    auto p = Line(m0, b0).point(x);
    p.y += distribution(generator);
    if (((N_FIFO) - (1)) < fifo.size()) {
      fifo.pop_back();
    }
    fifo.push_front(p);
  }

  auto m = (0.);
  auto b = (0.);
  ransac_line_fit(fifo, m, b);
  auto X = std::vector<double>();
  auto Y0 = std::vector<double>();
  auto Y1 = std::vector<double>();
  auto Y2 = std::vector<double>();
  for (auto i = 0; i < fifo.size(); i += 1) {
    auto x = fifo[i].x;
    auto p = Line(m0, b0).point(x);
    X.push_back(x);
    Y0.push_back(fifo[i].y);
    Y1.push_back(Line(m, b).point(x).y);
    Y2.push_back(p.y);
  }
  plt::named_plot("Y0", X, Y0);
  plt::named_plot("Y1", X, Y1);
  plt::named_plot("Y2", X, Y2);
  plt::show();
}
