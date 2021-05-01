#pragma once
#include "shared.h"

class Prior {
public:
  virtual double pdf(const Theta theta) const = 0;
  virtual Theta sample(Generator &rng) const = 0;
};

class UniformPrior : public Prior {

  size_t _d;
  std::vector<double> _lb, _ub;

public:
  UniformPrior(Bounds bounds) {
    _d = bounds.size();
    _lb.resize(_d);
    _ub.resize(_d);

    for (size_t i = 0; i < _d; i++) {
      _lb[i] = bounds[i][0];
      _ub[i] = bounds[i][1];
    }
  }

  double pdf(const Theta theta) const {
    for (size_t i = 0; i < _d; i++) {
      if (theta[i] < _lb[i] || theta[i] > _ub[i]) {
        return 0.0;
      }
    }
    return 1.0;
  }

  Theta sample(Generator &rng) const {
    Theta theta(_d);
    // TODO: Just use one uniform_real_distibution
    for (size_t i = 0; i < _d; i++) {
      std::uniform_real_distribution<double> U(_lb[i], _ub[i]);
      theta[i] = U(rng);
    }
    return theta;
  }
};