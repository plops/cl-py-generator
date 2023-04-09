// no preamble
#define FMT_HEADER_ONLY
#include "core.h"
extern "C" {
#include "esp_system.h"
#include "hardware.h"
#include "pax_codecs.h"
#include "pax_gfx.h"
};
#include "Display.h"
Display::Display() { pax_buf_init(&buf, nullptr, 320, 240, PAX_BUF_16_565RGB); }
void Display::background(uint8_t hue, uint8_t sat, uint8_t bright) {
  pax_background(&buf, pax_col_hsv(hue, sat, bright));
}
void Display::set_pixel(int x, int y, uint8_t hue, uint8_t sat,
                        uint8_t bright) {
  pax_set_pixel(&buf, pax_col_hsv(hue, sat, bright), x, y);
}
void Display::line(float x0, float y0, float x1, float y1, uint8_t h, uint8_t s,
                   uint8_t v) {
  pax_draw_line(&buf, pax_col_hsv(h, s, v), x0, y0, x1, y1);
}
void Display::small_text(std::string str, float x, float y, uint8_t h,
                         uint8_t s, uint8_t v) {
  text(str, pax_font_sky, x, y, h, s, v);
}
void Display::large_text(std::string str, float x, float y, uint8_t h,
                         uint8_t s, uint8_t v) {
  text(str, pax_font_saira_condensed, x, y, h, s, v);
}
void Display::text(std::string str, const pax_font_t *font, float x, float y,
                   uint8_t h, uint8_t s, uint8_t v) {
  auto text_ = str.c_str();
  auto dims = pax_text_size(font, font->default_size, text_);
  auto x_ = ((((buf.width) - (dims.x))) / ((2.0f)));
  auto y_ = ((((buf.height) - (dims.y))) / ((2.0f)));
  // center coordinate if x or y < 0
  if (0 < x) {
    x_ = x;
  }
  if (0 < y) {
    y_ = y;
  }
  pax_draw_text(&buf, pax_col_hsv(h, s, v), font, font->default_size, x_, y_,
                text_);
}
void Display::flush() {
  ili9341_write(get_ili9341(), static_cast<const uint8_t *>(buf.buf));
}
