#ifndef RANSAC_H
#define RANSAC_H

#include "DataTypes.h"
#include <deque>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

class Ransac  {
        private:
        double distance (Point2D p, double m, double b)       ;  
        void ransac_line_fit (std::deque<Point2D>& data, double& m, double& b, std::vector<Point2D>& inliers)       ;  
        std::vector<Point2D> m_inliers;
        std::deque<Point2D> m_data;
        double m_m;
        double m_b;
        public:
        double GetM ()       ;  
        double GetB ()       ;  
        std::vector<Point2D> GetInliers ()       ;  
        explicit  Ransac (std::deque<Point2D> data)       ;  
};

#endif /* !RANSAC_H */