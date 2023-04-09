// no preamble
#define FMT_HEADER_ONLY
#include "core.h"
#include <deque>

#include "Graph.h"
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
  auto text_ = fmt::format("T={:2.2f}°C", ((1.0f) * temperature));
  auto font = pax_font_saira_condensed;
  auto text = text_.c_str();
  auto dims = pax_text_size(font, font->default_size, text);
  m_display.small_text(text_, -1, (-10 + ((0.50f) * ((61.f) + 119))));

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
  auto text_ = fmt::format("H={:2.1f}%", ((1.0f) * humidity));
  auto font = pax_font_saira_condensed;
  auto text = text_.c_str();
  auto dims = pax_text_size(font, font->default_size, text);
  m_display.small_text(text_, -1, (-10 + ((0.50f) * ((121.f) + 179))));

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
  auto text_ = fmt::format("p={:4.2f}mbar", ((1.00e-2f) * pressure));
  auto font = pax_font_saira_condensed;
  auto text = text_.c_str();
  auto dims = pax_text_size(font, font->default_size, text);
  m_display.small_text(text_, -1, (-10 + ((0.50f) * ((181.f) + 239))));

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
