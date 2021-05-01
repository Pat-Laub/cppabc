#pragma once

#include "shared.h"

class KDE {
public:
  KDE() : _d(0) {}
  KDE(const WeightedSample &population);
  double pdf(const Theta theta) const;
  Theta sample(Generator &rng) const;

private:
  size_t _n, _d;
  Eigen::MatrixXd _samples;
  std::vector<double> _weights;
  std::vector<double> _weightsCDF;
  double _covDet;
  Eigen::MatrixXd _precision, _Q;
};
