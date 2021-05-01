#pragma once
#include "shared.h"

#pragma once
#include "shared.h"

Eigen::VectorXd weighted_mean(WeightedSample ws);
Eigen::MatrixXd weighted_covariance(WeightedSample ws);
double weighted_quantile(std::vector<double> data, std::vector<double> weights,
                         double quantile);

double effective_sample_size(std::vector<double> weights);