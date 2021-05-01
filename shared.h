#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>

#include <fmt/format.h>
#include <fmt/ranges.h>

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using Generator = std::minstd_rand;
using Theta = std::vector<double>;
using Sample = std::vector<double>;
using SumStats = std::vector<double>;
using Bounds = std::vector<std::array<double, 2>>;

struct WeightedSample {
  std::vector<Theta> samples;
  std::vector<double> weights;
};
