// no preamble
#include <algorithm>
#include <cmath>
#include <deque>
#include <random>
#include <vector>
#define FMT_HEADER_ONLY
#include "core.h"
const int RANSAC_MAX_ITERATIONS = 320;
const float RANSAC_INLIER_THRESHOLD = 5.0;
const int RANSAC_MIN_INLIERS = 2;
#include "Ransac.h"
double Ransac::distance(Point2D p, double m, double b) {
  return ((abs(((p.y) - (((((m) * (p.x))) + (b)))))) /
          (sqrt(((1) + (((m) * (m)))))));
}
void Ransac::ransac_line_fit(std::deque<Point2D> &data, double &m, double &b,
                             std::vector<Point2D> &inliers) {
  if (((data.size()) < (2))) {
    return;
  }
  std::random_device rd;
  // distrib0 must be one of the 5 most recent datapoints. i am not interested
  // in fit's of the older data

  auto gen = std::mt19937(rd());
  auto distrib0 = std::uniform_int_distribution<>(0, 5);
  auto distrib = std::uniform_int_distribution<>(0, ((data.size()) - (1)));
  auto best_inliers = std::vector<Point2D>();
  auto best_m = (0.);
  auto best_b = (0.);
  for (auto i = 0; (i) < (RANSAC_MAX_ITERATIONS); (i) += (1)) {
    auto idx1 = distrib(gen);
    auto idx2 = distrib0(gen);
    while ((idx1) == (idx2)) {
      idx1 = distrib(gen);
    }
    auto p1 = data[idx1];
    auto p2 = data[idx2];
    auto m = ((((p2.y) - (p1.y))) / (((p2.x) - (p1.x))));
    auto b = ((p1.y) - (((m) * (p1.x))));
    auto inliers = std::vector<Point2D>();
    for (auto &p : data) {
      if (((distance(p, m, b)) < (RANSAC_INLIER_THRESHOLD))) {
        inliers.push_back(p);
      }
    };
    if (((RANSAC_MIN_INLIERS) < (inliers.size()))) {
      auto sum_x = (0.);
      auto sum_y = (0.);
      for (auto &p : inliers) {
        (sum_x) += (p.x);
        (sum_y) += (p.y);
      };
      auto avg_x = ((sum_x) / (inliers.size()));
      auto avg_y = ((sum_y) / (inliers.size()));
      auto var_x = (0.);
      auto cov_xy = (0.);
      for (auto &p : inliers) {
        (var_x) += (((((p.x) - (avg_x))) * (((p.x) - (avg_x)))));
        (cov_xy) += (((((p.x) - (avg_x))) * (((p.y) - (avg_y)))));
      };
      auto m = ((cov_xy) / (var_x));
      auto b = ((avg_y) - (((m) * (avg_x))));
      if (((best_inliers.size()) < (inliers.size()))) {
        best_inliers = inliers;
        best_m = m;
        best_b = b;
      }
    }
  }
  m = best_m;
  b = best_b;
  inliers = best_inliers;
}
double Ransac::GetM() const { return m_m; }
double Ransac::GetB() const { return m_b; }
std::vector<Point2D> Ransac::GetInliers() { return m_inliers; }
Ransac::Ransac(std::deque<Point2D> data)
    : m_data(data) {
  ransac_line_fit(m_data, m_m, m_b, m_inliers);
}
