#ifndef BMESENSOR_H
#define BMESENSOR_H

#include "DataTypes.h"
#include <deque>
extern "C"  {
            #include "bme680.h"

};

class BmeSensor  {
        private:
        static constexpr char*TAG = "mch2022-co2-bme";
        public:
        void measureBME (std::deque<PointBME>& fifoBME, std::deque<Point2D>& fifo)       ;  
        explicit  BmeSensor ()       ;  
};

#endif /* !BMESENSOR_H */