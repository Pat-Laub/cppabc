#include "weighted.h"

double weighted_quantile(std::vector<double> data, std::vector<double> weights,
                         double quantile) {

  int n = (int)data.size();

  std::vector<size_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);

  std::sort(idx.begin(), idx.end(),
            [&data](size_t i1, size_t i2) { return data[i1] < data[i2]; });

  std::vector<double> sortedWeights(n);
  double cusumWeights = 0.0;
  for (int i = 0; i < n; i++) {
    cusumWeights += weights[idx[i]];
    if (cusumWeights > quantile) {
      return data[idx[i]];
    }
  }
  return data[idx[n - 1]];
}

Eigen::VectorXd weighted_mean(WeightedSample ws) {

  Eigen::Map<Eigen::VectorXd> weights(ws.weights.data(), ws.weights.size());

  Eigen::MatrixXd X(ws.samples.size(), ws.samples[0].size());
  for (int i = 0; i < X.rows(); i++) {
    for (int j = 0; j < X.cols(); j++) {
      X(i, j) = ws.samples[i][j];
    }
  }

  Eigen::VectorXd mean(X.cols());
  for (int j = 0; j < X.cols(); j++) {
    mean[j] = (weights.array() * X.col(j).array()).mean();
  }

  return mean;
}

Eigen::MatrixXd weighted_covariance(WeightedSample ws) {

  Eigen::Map<Eigen::VectorXd> weights(ws.weights.data(), ws.weights.size());

  Eigen::MatrixXd X(ws.samples.size(), ws.samples[0].size());
  for (int i = 0; i < X.rows(); i++) {
    for (int j = 0; j < X.cols(); j++) {
      X(i, j) = ws.samples[i][j];
    }
  }

  Eigen::MatrixXd centered = X.rowwise() - weighted_mean(ws).transpose();
  return (centered.adjoint() * centered) / double(X.rows() - 1);
}

double effective_sample_size(std::vector<double> weights) {
  Eigen::Map<Eigen::ArrayXd> w(weights.data(), weights.size());
  return 1.0 / (w.pow(2).sum());
}