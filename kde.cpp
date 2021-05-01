#include "kde.h"
#include "weighted.h"

const double PI = 3.141592653589793238463;

KDE::KDE(const WeightedSample &population) {

  _n = population.samples.size();
  _d = population.samples[0].size();

  _samples = Eigen::MatrixXd(_n, _d);
  for (size_t i = 0; i < _n; i++) {
    for (size_t j = 0; j < _d; j++) {
      _samples(i, j) = population.samples[i][j];
    }
  }

  _weights = population.weights;
  _weightsCDF = std::vector<double>(_weights.size());
  std::partial_sum(_weights.begin(), _weights.end(), _weightsCDF.begin());

  Eigen::MatrixXd cov = weighted_covariance(population);

  double bw = 1.0 / _n;
  cov *= bw;

  _precision = cov.inverse();
  _covDet = cov.determinant();

  // Find the eigen vectors of the covariance matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(cov);
  Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors().real();

  // Find the eigenvalues of the covariance matrix
  Eigen::MatrixXd eigenvalues = eigen_solver.eigenvalues().real().asDiagonal();

  // Find the transformation matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(eigenvalues);
  Eigen::MatrixXd sqrt_eigenvalues = es.operatorSqrt();
  _Q = eigenvectors * sqrt_eigenvalues;
}

double KDE::pdf(const Theta theta) const {
  // MVN code adapted from http://blog.sarantop.com/notes/mvn
  double pdf = 0.0;
  const Eigen::Map<const Eigen::VectorXd> x(theta.data(), _d);
  const double sqrt2pi = std::sqrt(2 * PI);
  const double norm = std::pow(sqrt2pi, -(int)_d) * std::pow(_covDet, -0.5);
  const double lognorm = std::log(norm);

  for (size_t i = 0; i < _n; i++) {
    double quadform = (x - _samples.row(i).transpose()).transpose() *
                      _precision * (x - _samples.row(i).transpose());
    pdf += std::exp(std::log(_weights[i]) + lognorm - 0.5 * quadform);
  }

  assert(pdf > 0);

  return pdf;
}

Theta KDE::sample(Generator &rng) const {

  std::normal_distribution<double> N;
  Eigen::VectorXd Z(_d);

  for (size_t i = 0; i < _d; i++) {
    Z[i] = N(rng);
  }

  // TODO: Compare to making new discrete distibution object here.
  std::uniform_real_distribution<double> U(0.0, 1.0);
  double u = U(rng);
  int p = _n - 1;
  for (size_t i = 0; i < _n - 1; i++) {
    if (u < _weightsCDF[i]) {
      p = i;
      break;
    }
  }

  Eigen::VectorXd X = _samples.row(p).transpose() + _Q * Z;

  Theta theta(_d);
  for (size_t i = 0; i < _d; i++) {
    theta[i] = X[i];
  }

  return theta;
}