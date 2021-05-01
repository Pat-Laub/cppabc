#include "kde.h"
#include "prior.h"
#include "shared.h"
#include "weighted.h"

#include <limits>
#include <omp.h>

SumStats get_summary_statistics(Sample &x) {
  std::sort(x.begin(), x.end());
  return std::move(x);
}

double distance(const SumStats &x, const SumStats &y) {

  double dist = 0.0;
  for (int i = 0; i < x.size(); i++) {
    dist += std::abs(x[i] - y[i]);
    if ((x[i] == 0) || (y[i] == 0)) {
      if ((x[i] != 0) || (y[i] != 0)) {
        return std::numeric_limits<double>::infinity();
      }
    }
  }

  return dist / x.size();
}

Sample sample_x(Generator &rng, int T, const Theta &theta) {

  double p = theta[0];
  double delta = theta[1];

  std::geometric_distribution<int> N(1 - p);
  std::exponential_distribution<double> U(1 / delta);

  std::vector<double> x(T);
  for (int s = 0; s < T; s++) {
    int n_s = N(rng);

    // Assume sum aggregator for now
    x[s] = 0.0;
    for (int k = 0; k < n_s; k++) {
      x[s] += U(rng);
    }
  }
  return std::move(x);
}

double safe_divide(double d1, double d2) {
  return std::exp(std::log(d1) - std::log(d2));
}

std::tuple<Theta, double, double>
sample_one(const Prior &prior, const SumStats &obsSS, const KDE &kde, int seed,
           int iter, int T, double epsilon, int maxAttempts = 1e9) {

  Generator rng(seed);

  if (iter == 0) {

    for (int attempt = 0; attempt < maxAttempts; attempt++) {
      Theta guess = prior.sample(rng);
      Sample fakeData = sample_x(rng, T, guess);
      SumStats fakeSS = get_summary_statistics(fakeData);

      double d = distance(obsSS, fakeSS);
      if (d < epsilon) {
        return {guess, 1.0, d};
      }
    }

    return {};

  } else {
    for (int attempt = 0; attempt < maxAttempts; attempt++) {
      Theta guess = kde.sample(rng);

      double priorPDF = prior.pdf(guess);
      if (priorPDF == 0.0) {
        continue;
      }

      Sample fakeData = sample_x(rng, T, guess);
      SumStats fakeSS = get_summary_statistics(fakeData);

      double d = distance(obsSS, fakeSS);
      if (d < epsilon) {
        double weight = safe_divide(priorPDF, kde.pdf(guess));

        return {guess, weight, d};
      }
    }

    return {};
  }
}

std::pair<WeightedSample, std::vector<double>>
sample_population(const SumStats &obsSS, const Prior &prior, const KDE &kde,
                  int iter, int T, int popSize, double epsilon) {

  std::vector<Theta> samples(popSize);
  std::vector<double> weights(popSize);
  std::vector<double> dists(popSize);

  int p;
#pragma omp parallel for
  for (p = 0; p < popSize; p++) {
    auto [sample, weight, d] =
        sample_one(prior, obsSS, kde, p, iter, T, epsilon);
    samples[p] = sample;
    weights[p] = weight;
    dists[p] = d;
  }

  double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);

  for (p = 0; p < popSize; p++) {
    weights[p] /= weightSum;
  }

  return {{samples, weights}, dists};
}

std::vector<Theta> acceptance_rejection(Sample obs, const Prior &prior,
                                        int popSize, double epsilon,
                                        int seed = 42) {

  auto start = std::chrono::high_resolution_clock::now();

  int T = (int)obs.size();
  SumStats obsSS = get_summary_statistics(obs);

  auto [population, dists] =
      sample_population(obsSS, prior, {}, 0, T, popSize, epsilon);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  fmt::print("ABC-AR sampling took {} secs\n", elapsed.count());

  return population.samples;
}

WeightedSample sequential_monte_carlo(Sample obs, const Prior &prior,
                                      int numIters, int popSize, int seed = 42,
                                      double quantile = 0.5) {

  auto start = std::chrono::high_resolution_clock::now();

  int T = (int)obs.size();
  SumStats obsSS = get_summary_statistics(obs);

  WeightedSample population;
  std::vector<double> dists;
  double epsilon;
  KDE kde;
  std::chrono::duration<double> elapsed, iterElapsed;

  for (int iter = 0; iter < numIters; iter++) {
    if (iter == 0) {
      epsilon = std::numeric_limits<double>::max();
      fmt::print("Iteration 1; Sampling with d(obs, fake) <= Inf; ");
    } else {
      kde = KDE(population);
      epsilon = weighted_quantile(dists, population.weights, quantile);
      fmt::print("Iteration {}; Sampling with d(obs, fake) <= {:.2g}; ",
                 iter + 1, epsilon);
    }

    auto iterStart = std::chrono::high_resolution_clock::now();

    std::tie(population, dists) =
        sample_population(obsSS, prior, kde, iter, T, popSize, epsilon);

    auto iterEnd = std::chrono::high_resolution_clock::now();
    iterElapsed = iterEnd - iterStart;
    fmt::print("Took {:.4f} secs; ", iterElapsed.count());

    double ESS = effective_sample_size(population.weights);
    fmt::print("ESS is {:.2f}\n", ESS);
  }

  auto end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  fmt::print("ABC-SMC sampling took {} secs\n", elapsed.count());

  Eigen::Map<Eigen::VectorXd> d(dists.data(), popSize);
  fmt::print("All samples have d(obs, fake) <= {}\n", d.maxCoeff());

  return population;
}

int main(int argc, char *argv[]) {

  try {
    int numIters = 20, popSize = 100;
    if (argc > 1) {
      numIters = atoi(argv[1]);
    }
    if (argc > 2) {
      popSize = atoi(argv[2]);
    }

    omp_set_num_threads(4);

    int seed = 42;
    double quantile = 0.5;

    const UniformPrior prior = Bounds{{0.0, 1.0}, {0.0, 100.0}};

    int T = 100;
    Theta thetaTrue = {0.8, 5.0};

    Generator rng(1337);
    Sample obs = sample_x(rng, T, thetaTrue);

    // std::vector<Theta> population =
    //     acceptance_rejection(obs, sample_prior, popSize, epsilon);

    // int actualPopSize = 0;
    // double pMean = 0.0, deltaMean = 0.0;
    // for (int i = 0; i < popSize; i++) {
    //   if (population[i].size() > 0) {
    //     pMean += population[i][0];
    //     deltaMean += population[i][1];
    //     actualPopSize += 1;
    //   }
    // }
    // pMean /= popSize;
    // deltaMean /= popSize;

    // std::cout << "Actual population size is " << actualPopSize <<
    // std::endl; std::cout << "p mean is " << pMean << std::endl; std::cout
    // << "delta mean is " << deltaMean << std::endl;

    WeightedSample ws =
        sequential_monte_carlo(obs, prior, numIters, popSize, seed, quantile);
    int actualPopSize = 0;
    double pMean = 0.0, deltaMean = 0.0;
    for (int i = 0; i < popSize; i++) {
      if (ws.samples[i].size() > 0) {
        pMean += ws.weights[i] * ws.samples[i][0];
        deltaMean += ws.weights[i] * ws.samples[i][1];
        actualPopSize += 1;
      }
    }

    std::cout << "Actual population size is " << actualPopSize << std::endl;
    std::cout << "p mean is " << pMean << std::endl;
    std::cout << "delta mean is " << deltaMean << std::endl;

    return 0;

  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    ;
  } catch (...) {
    std::cerr << "Unknown error" << std::endl;
  }
  return 1;
}
