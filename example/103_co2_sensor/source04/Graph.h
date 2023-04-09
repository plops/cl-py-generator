#ifndef GRAPH_H
#define GRAPH_H

#include "DataTypes.h"
#include "Display.h"
#include <deque>

class Graph  {
        private:
        Display& m_display;
        std::deque<Point2D>& m_fifo; 
        std::deque<PointBME>& m_fifoBME; 
        public:
        void carbon_dioxide ()       ;  
        void temperature ()       ;  
        void humidity ()       ;  
        void pressure ()       ;  
        explicit  Graph (Display& display, std::deque<Point2D>& fifo, std::deque<PointBME>& fifoBME)       ;  
};

#endif /* !GRAPH_H */