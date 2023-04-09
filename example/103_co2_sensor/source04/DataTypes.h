#pragma once
struct Point2D {
  double x;
  double y;
};
typedef struct Point2D Point2D;

struct PointBME {
  double x;
  double temperature;
  double humidity;
  double pressure;
};
typedef struct PointBME PointBME;
