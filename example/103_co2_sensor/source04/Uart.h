#ifndef UART_H
#define UART_H

#include "DataTypes.h"
#include <deque>
extern "C"  {
            #include "driver/uart.h"

};

class Uart  {
        private:
        static constexpr char*TAG = "mch2022-co2-uart";
        static constexpr uart_port_t CO2_UART=UART_NUM_1;
        static constexpr size_t BUF_SIZE=UART_FIFO_LEN;
        public:
        void measureCO2 (std::deque<Point2D>& fifo)       ;  
        explicit  Uart ()       ;  
};

#endif /* !UART_H */