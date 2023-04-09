#ifndef DISPLAY_H
#define DISPLAY_H

#include <string>
extern "C"  {
            #include "pax_gfx.h"
#include "pax_codecs.h"

};

class Display  {
        private:
        pax_buf_t buf;
        public:
        explicit  Display ()       ;  
        void background (uint8_t hue, uint8_t sat, uint8_t bright)       ;  
        void set_pixel (int x, int y, uint8_t hue = 128, uint8_t sat = 0, uint8_t bright = 0)       ;  
        void small_text (std::string text, float x = -1.0f, float y = -1.0f, uint8_t h = 128, uint8_t s = 255, uint8_t v = 255)       ;  
        void large_text (std::string text, float x = -1.0f, float y = -1.0f, uint8_t h = 128, uint8_t s = 255, uint8_t v = 255)       ;  
        void text (std::string text, pax_font_t* font, float x, float y, uint8_t h, uint8_t s, uint8_t v)       ;  
        void flush ()       ;  
};

#endif /* !DISPLAY_H */