// no preamble
#define FMT_HEADER_ONLY
#include "Ransac.h"
#include "core.h"
#include <deque>

#include "Graph.h"
void Graph::carbon_dioxide() {
  if (m_fifo.size() < 2) {
    return;
  }
  auto hue = 12;
  auto sat = 255;
  auto bright = 255;
  auto col = pax_col_hsv(hue, sat, bright);
  auto time_ma = m_fifo[0].x;
  auto time_mi = m_fifo[((m_fifo.size()) - (1))].x;
  auto time_delta = ((time_ma) - (time_mi));
  auto scaleTime = [&](float x) -> float {
    auto res = ((318.f) * ((((x) - (time_mi))) / (time_delta)));
    if (res < (1.0f)) {
      res = (1.0f);
    }
    if ((318.f) < res) {
      res = (318.f);
    }
    return res;
  };
  auto min_max_y = std::minmax_element(
      m_fifo.begin(), m_fifo.end(),
      [](const Point2D &p1, const Point2D &p2) { return p1.y < p2.y; });
  auto min_y = min_max_y.first->y;
  auto max_y = min_max_y.second->y;
  auto scaleHeight = [&](float v) -> float {
    // v is in the range 400 .. 5000
    // map to 0 .. 239

    auto mi = (4.00e+2f);
    auto ma = (max_y < (1.20e+3f)) ? ((1.20e+3f)) : ((5.00e+3f));
    auto res =
        ((1.0f) + (59 * (((1.0f)) - (((((v) - (mi))) / (((ma) - (mi))))))));
    if (res < (1.0f)) {
      res = (1.0f);
    }
    if (59 < res) {
      res = 59;
    }
    return res;
  };
  // write latest measurement
  auto co2 = m_fifo[0].y;
  m_display.large_text(fmt::format("CO2={:4.0f}ppm", co2), -1,
                       (-10 + ((0.50f) * ((1.0f) + 59))));

  for (auto p : m_fifo) {
    // draw measurements as points

    for (auto i = 0; i < 3; i += 1) {
      for (auto j = 0; j < 3; j += 1) {
        m_display.set_pixel((i + -1 + scaleTime(p.x)),
                            (j + -1 + scaleHeight(p.y)), 149, 180, 200);
      }
    }
  }
  {
    auto ransac = Ransac(m_fifo);
    auto m = ransac.GetM();
    auto b = ransac.GetB();
    auto inliers = ransac.GetInliers();
    auto hue = 128;
    auto sat = 255;
    auto bright = 200;
    auto col = pax_col_hsv(hue, sat, bright);
    // draw the fit as line
    m_display.line(scaleTime(time_mi), scaleHeight((b + (m * time_mi))),
                   scaleTime(time_ma), scaleHeight((b + (m * time_ma))), 188,
                   255, 200);
    // draw inliers as points
    for (auto p : inliers) {
      for (auto i = 0; i < 3; i += 1) {
        for (auto j = 0; j < 3; j += 1) {
          m_display.set_pixel((i + -1 + scaleTime(p.x)),
                              (j + -1 + scaleHeight(p.y)), 0, 255, 255);
        }
      }
    }

    // compute when a value of 1200ppm is reached
    auto x0 = (((((1.20e+3)) - (b))) / (m));
    auto x0l = (((((5.00e+2)) - (b))) / (m));
    m_display.small_text(
        fmt::format("m={:3.4f} b={:4.2f} xmi={:4.2f} xma={:4.2f}", m, b,
                    time_mi, time_ma),
        20, 80, 160, 128, 128);
    m_display.small_text(fmt::format("x0={:4.2f} x0l={:4.2f}", x0, x0l), 20, 60,
                         130, 128, 128);
    if (time_ma < x0) {
      // if predicted intersection time is in the future, print it
      auto time_value = static_cast<int>(((x0) - (time_ma)));
      auto hours = int(((time_value) / (3600)));
      auto minutes = int(((time_value % 3600) / (60)));
      auto seconds = time_value % 60;
      auto text_ = fmt::format("air room in (h:m:s) {:02d}:{:02d}:{:02d}",
                               hours, minutes, seconds);
      m_display.small_text(text_, 20, 140, 30, 128, 128);

    } else {
      // if predicted intersection time is in the past, then predict when airing
      // should stop
      auto x0 = (((((5.00e+2)) - (b))) / (m));
      auto time_value = static_cast<int>(((x0) - (time_ma)));
      auto hours = int(((time_value) / (3600)));
      auto minutes = int(((time_value % 3600) / (60)));
      auto seconds = time_value % 60;
      auto text_ =
          fmt::format("air of room should stop in (h:m:s) {:02d}:{:02d}:{:02d}",
                      hours, minutes, seconds);
      m_display.small_text(text_, 20, 140, 90, 128, 128);
    }
  }
}
void Graph::temperature() {
  auto time_ma = m_fifoBME[0].x;
  auto time_mi = m_fifoBME[((m_fifoBME.size()) - (1))].x;
  auto time_delta = ((time_ma) - (time_mi));
  auto scaleTime = [&](float x) -> float {
    auto res = ((318.f) * ((((x) - (time_mi))) / (time_delta)));
    if (res < (1.0f)) {
      res = (1.0f);
    }
    if ((318.f) < res) {
      res = (318.f);
    }
    return res;
  };
  auto min_max_y =
      std::minmax_element(m_fifoBME.begin(), m_fifoBME.end(),
                          [](const PointBME &p1, const PointBME &p2) {
                            return p1.temperature < p2.temperature;
                          });
  auto min_y = min_max_y.first->temperature;
  auto max_y = min_max_y.second->temperature;
  auto scaleHeight = [&](float v) -> float {
    auto mi = min_y;
    auto ma = max_y;
    auto res =
        ((61.f) + (59 * (((1.0f)) - (((((v) - (mi))) / (((ma) - (mi))))))));
    if (res < (61.f)) {
      res = (61.f);
    }
    if (119 < res) {
      res = 119;
    }
    return res;
  };
  // write latest measurement
  auto temperature = m_fifoBME[0].temperature;
  m_display.large_text(fmt::format("T={:2.2f}°C", ((1.0f) * temperature)), -1,
                       (-10 + ((0.50f) * ((61.f) + 119))));

  for (auto p : m_fifoBME) {
    // draw measurements as points

    for (auto i = 0; i < 3; i += 1) {
      for (auto j = 0; j < 3; j += 1) {
        m_display.set_pixel((i + -1 + scaleTime(p.x)),
                            (j + -1 + scaleHeight(p.temperature)), 150, 180,
                            200);
      }
    }
  }
}
void Graph::humidity() {
  auto time_ma = m_fifoBME[0].x;
  auto time_mi = m_fifoBME[((m_fifoBME.size()) - (1))].x;
  auto time_delta = ((time_ma) - (time_mi));
  auto scaleTime = [&](float x) -> float {
    auto res = ((318.f) * ((((x) - (time_mi))) / (time_delta)));
    if (res < (1.0f)) {
      res = (1.0f);
    }
    if ((318.f) < res) {
      res = (318.f);
    }
    return res;
  };
  auto min_max_y =
      std::minmax_element(m_fifoBME.begin(), m_fifoBME.end(),
                          [](const PointBME &p1, const PointBME &p2) {
                            return p1.humidity < p2.humidity;
                          });
  auto min_y = min_max_y.first->humidity;
  auto max_y = min_max_y.second->humidity;
  auto scaleHeight = [&](float v) -> float {
    auto mi = min_y;
    auto ma = max_y;
    auto res =
        ((121.f) + (59 * (((1.0f)) - (((((v) - (mi))) / (((ma) - (mi))))))));
    if (res < (121.f)) {
      res = (121.f);
    }
    if (179 < res) {
      res = 179;
    }
    return res;
  };
  // write latest measurement
  auto humidity = m_fifoBME[0].humidity;
  m_display.large_text(fmt::format("H={:2.1f}%", ((1.0f) * humidity)), -1,
                       (-10 + ((0.50f) * ((121.f) + 179))));

  for (auto p : m_fifoBME) {
    // draw measurements as points

    for (auto i = 0; i < 3; i += 1) {
      for (auto j = 0; j < 3; j += 1) {
        m_display.set_pixel((i + -1 + scaleTime(p.x)),
                            (j + -1 + scaleHeight(p.humidity)), 80, 180, 200);
      }
    }
  }
}
void Graph::pressure() {
  auto time_ma = m_fifoBME[0].x;
  auto time_mi = m_fifoBME[((m_fifoBME.size()) - (1))].x;
  auto time_delta = ((time_ma) - (time_mi));
  auto scaleTime = [&](float x) -> float {
    auto res = ((318.f) * ((((x) - (time_mi))) / (time_delta)));
    if (res < (1.0f)) {
      res = (1.0f);
    }
    if ((318.f) < res) {
      res = (318.f);
    }
    return res;
  };
  auto min_max_y =
      std::minmax_element(m_fifoBME.begin(), m_fifoBME.end(),
                          [](const PointBME &p1, const PointBME &p2) {
                            return p1.pressure < p2.pressure;
                          });
  auto min_y = min_max_y.first->pressure;
  auto max_y = min_max_y.second->pressure;
  auto scaleHeight = [&](float v) -> float {
    auto mi = min_y;
    auto ma = max_y;
    auto res =
        ((181.f) + (59 * (((1.0f)) - (((((v) - (mi))) / (((ma) - (mi))))))));
    if (res < (181.f)) {
      res = (181.f);
    }
    if (239 < res) {
      res = 239;
    }
    return res;
  };
  // write latest measurement
  auto pressure = m_fifoBME[0].pressure;
  m_display.large_text(fmt::format("p={:4.2f}mbar", ((1.00e-2f) * pressure)),
                       -1, (-10 + ((0.50f) * ((181.f) + 239))));

  for (auto p : m_fifoBME) {
    // draw measurements as points

    for (auto i = 0; i < 3; i += 1) {
      for (auto j = 0; j < 3; j += 1) {
        m_display.set_pixel((i + -1 + scaleTime(p.x)),
                            (j + -1 + scaleHeight(p.pressure)), 240, 180, 200);
      }
    }
  }
}
Graph::Graph(Display &display, std::deque<Point2D> &fifo,
             std::deque<PointBME> &fifoBME)
    : m_display(display), m_fifo(fifo), m_fifoBME(fifoBME) {}
