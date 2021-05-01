#include <benchmark/benchmark.h>

#ifdef _WIN32
#pragma comment(lib, "Shlwapi.lib")
#ifdef _DEBUG
#pragma comment(lib, "benchmarkd.lib")
#else
#pragma comment(lib, "benchmark.lib")
#endif
#endif

#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include <random>

static void bm_sqrt(benchmark::State &state) {
  state.SetLabel("'sqrt' function");

  double i = 0.0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::sqrt(i));
    i += 1.0;
  }
}

BENCHMARK(bm_sqrt);

static void bm_pow_half(benchmark::State &state) {
  state.SetLabel("'pow(., 0.5)' function");

  double i = 0.0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::pow(i, 0.5));
    i += 1.0;
  }
}

BENCHMARK(bm_pow_half);

using Generator = std::minstd_rand;
using Theta = std::vector<double>;
using Sample = std::vector<double>;
using SumStats = std::vector<double>;

// TODO:

//  Theta sample(Generator &rng) const {
//     Theta theta(_d);
//     for (size_t i = 0; i < _d; i++) {
//       std::uniform_real_distribution<double> U(_lb[i], _ub[i]);
//       theta[i] = U(rng);
//     }
//     return theta;
//   }

  // Theta sample(Generator &rng) {
  //   std::uniform_real_distribution<double> U(0.0, 1.0);
  //   Theta theta(_d);
  //   for (int i = 0; i < _d; i++) {
  //     theta[i] = _lb[i] + U(rng) * (_ub[i] - _lb[i]);
  //   }
  //   return theta;
  // }

SumStats get_summary_statistics(Sample &x) {
  std::sort(x.begin(), x.end());
  return std::move(x);
}

SumStats get_summary_statistics_no_ref_move(Sample x) {
  std::sort(x.begin(), x.end());
  return std::move(x);
}

SumStats get_summary_statistics_no_ref_no_move(Sample x) {
  std::sort(x.begin(), x.end());
  return x;
}

SumStats get_summary_statistics_no_move(Sample &x) {
  std::sort(x.begin(), x.end());
  return x;
}

double distance(const SumStats &x, const SumStats &y) {

  double dist = 0.0;
  for (int i = 0; i < x.size(); i++) {
    dist += std::abs(y[i] - x[i]);
  }

  return dist / x.size();
}

Sample sample_theta(Generator &rng, int t, const Theta &theta) {

  double p = theta[0];
  double delta = theta[1];

  std::geometric_distribution<int> Geom(1 - p);
  std::exponential_distribution<double> Exp(1 / delta);

  std::vector<double> x(t);
  for (int s = 0; s < t; s++) {
    int n_s = Geom(rng);

    // Assume sum aggregator for now
    x[s] = 0.0;
    for (int k = 0; k < n_s; k++) {
      x[s] += Exp(rng);
    }
  }
  return std::move(x);
}

Theta sample_one(int seed, Theta sample_prior(std::minstd_rand &),
                 SumStats obsSS, int t, double epsilon, int maxAttempts = 1e9) {

  Generator rng(seed);

  for (int attempt = 0; attempt < maxAttempts; attempt++) {
    Theta guess = sample_prior(rng);

    Sample fakeData = sample_theta(rng, t, guess);
    SumStats fakeSS = get_summary_statistics(fakeData);

    double d = distance(obsSS, fakeSS);
    if (d < epsilon) {
      return guess;
    }
  }

  return {};
}

static void bm_sort(benchmark::State &state) {

  int t = 100;
  double pTrue = 0.8, deltaTrue = 5.0;
  Theta thetaTrue = {0.8, 2.0};

  Generator rng(1337);
  Sample obsData = sample_theta(rng, t, thetaTrue);

  for (auto _ : state) {
    state.PauseTiming();
    std::shuffle(obsData.begin(), obsData.end(), rng);
    state.ResumeTiming();

    std::sort(obsData.begin(), obsData.end());
    assert(obsData[0] < obsData[1]);
  }
}

// BENCHMARK(bm_sort);

static void bm_sort_vector(benchmark::State &state) {

  int t = 100;
  double pTrue = 0.8, deltaTrue = 5.0;
  Theta thetaTrue = {0.8, 2.0};

  Generator rng(1337);
  std::vector<double> obsData = sample_theta(rng, t, thetaTrue);

  for (auto _ : state) {
    state.PauseTiming();
    std::shuffle(obsData.begin(), obsData.end(), rng);
    state.ResumeTiming();

    std::sort(obsData.begin(), obsData.end());
    assert(obsData[0] < obsData[1]);
  }
}

// BENCHMARK(bm_sort_vector);

static void bm_sort_std(benchmark::State &state) {

  int t = 100;
  double pTrue = 0.8, deltaTrue = 5.0;
  Theta thetaTrue = {0.8, 2.0};

  Generator rng(1337);
  std::vector<double> obsData = sample_theta(rng, t, thetaTrue);

  for (auto _ : state) {
    state.PauseTiming();
    std::shuffle(obsData.begin(), obsData.end(), rng);
    state.ResumeTiming();

    std::sort(std::begin(obsData), std::end(obsData));
    assert(obsData[0] < obsData[1]);
  }
}

// BENCHMARK(bm_sort_std);

static void bm_ss(benchmark::State &state) {

  int t = 100;
  double pTrue = 0.8, deltaTrue = 5.0;
  Theta thetaTrue = {0.8, 2.0};

  Generator rng(1337);
  Sample obsData = sample_theta(rng, t, thetaTrue);

  for (auto _ : state) {
    state.PauseTiming();
    std::shuffle(obsData.begin(), obsData.end(), rng);
    state.ResumeTiming();

    obsData = get_summary_statistics(obsData);
    assert(obsData[0] < obsData[1]);
  }
}

BENCHMARK(bm_ss);

static void bm_ss_no_move(benchmark::State &state) {

  int t = 100;
  double pTrue = 0.8, deltaTrue = 5.0;
  Theta thetaTrue = {0.8, 2.0};

  Generator rng(1337);
  Sample obsData = sample_theta(rng, t, thetaTrue);

  for (auto _ : state) {
    state.PauseTiming();
    std::shuffle(obsData.begin(), obsData.end(), rng);
    state.ResumeTiming();

    SumStats obsSS = get_summary_statistics_no_move(obsData);
    assert(obsSS[0] < obsSS[1]);
  }
}

// BENCHMARK(bm_ss_no_move);

static void bm_ss_no_ref_move(benchmark::State &state) {

  int t = 100;
  double pTrue = 0.8, deltaTrue = 5.0;
  Theta thetaTrue = {0.8, 2.0};

  Generator rng(1337);
  Sample obsData = sample_theta(rng, t, thetaTrue);

  for (auto _ : state) {
    state.PauseTiming();
    std::shuffle(obsData.begin(), obsData.end(), rng);
    state.ResumeTiming();

    SumStats obsSS = get_summary_statistics_no_ref_move(obsData);
    assert(obsSS[0] < obsSS[1]);
  }
}

// BENCHMARK(bm_ss_no_ref_move);

static void bm_ss_no_ref_no_move(benchmark::State &state) {

  int t = 100;
  double pTrue = 0.8, deltaTrue = 5.0;
  Theta thetaTrue = {0.8, 2.0};

  Generator rng(1337);
  Sample obsData = sample_theta(rng, t, thetaTrue);

  for (auto _ : state) {
    state.PauseTiming();
    std::shuffle(obsData.begin(), obsData.end(), rng);
    state.ResumeTiming();

    SumStats obsSS = get_summary_statistics_no_ref_no_move(obsData);
    assert(obsSS[0] < obsSS[1]);
  }
}

// BENCHMARK(bm_ss_no_ref_no_move);

static void bm_sample_one_lambda(benchmark::State &state) {
  //   state.SetLabel("'pow(., 0.5)' function");
  double epsilon = 1.0;

  int t = 100;
  double pTrue = 0.8, deltaTrue = 5.0;
  Theta thetaTrue = {0.8, 2.0};

  Generator rng(1337);
  Sample obsData = sample_theta(rng, t, thetaTrue);
  SumStats obsSS = get_summary_statistics(obsData);

  auto sample_prior = [](Generator &rng) -> Theta {
    std::uniform_real_distribution<double> pPrior(0.0, 1.0);
    std::uniform_real_distribution<double> deltaPrior(0.0, 100.0);
    return {pPrior(rng), deltaPrior(rng)};
  };

  int seed = 0;
  for (auto _ : state) {
    sample_one(seed, sample_prior, obsSS, t, epsilon);
  }
}

// BENCHMARK(bm_sample_one_lambda)->Unit(benchmark::kMicrosecond);

static void bm_sample_one_lambda_const(benchmark::State &state) {
  //   state.SetLabel("'pow(., 0.5)' function");
  double epsilon = 1.0;

  int t = 100;
  double pTrue = 0.8, deltaTrue = 5.0;
  Theta thetaTrue = {0.8, 2.0};

  Generator rng(1337);
  Sample obsData = sample_theta(rng, t, thetaTrue);
  SumStats obsSS = get_summary_statistics(obsData);

  auto sample_prior = [](Generator &rng) -> Theta {
    const std::uniform_real_distribution<double> pPrior(0.0, 1.0);
    const std::uniform_real_distribution<double> deltaPrior(0.0, 100.0);
    return {pPrior(rng), deltaPrior(rng)};
  };

  int seed = 0;
  for (auto _ : state) {
    sample_one(seed, sample_prior, obsSS, t, epsilon);
  }
}

BENCHMARK(bm_sample_one_lambda_const)->Unit(benchmark::kMicrosecond);

static void bm_sample_one_lambda_static(benchmark::State &state) {
  //   state.SetLabel("'pow(., 0.5)' function");
  double epsilon = 1.0;

  int t = 100;
  double pTrue = 0.8, deltaTrue = 5.0;
  Theta thetaTrue = {0.8, 2.0};

  Generator rng(1337);
  Sample obsData = sample_theta(rng, t, thetaTrue);
  SumStats obsSS = get_summary_statistics(obsData);

  auto sample_prior = [](Generator &rng) -> Theta {
    static const std::uniform_real_distribution<double> pPrior(0.0, 1.0);
    static const std::uniform_real_distribution<double> deltaPrior(0.0, 100.0);
    return {pPrior(rng), deltaPrior(rng)};
  };

  int seed = 0;
  for (auto _ : state) {
    sample_one(seed, sample_prior, obsSS, t, epsilon);
  }
}

// BENCHMARK(bm_sample_one_lambda_static)->Unit(benchmark::kMicrosecond);

Theta sample_prior_function(Generator &rng) {
  const std::uniform_real_distribution<double> pPrior(0.0, 1.0);
  const std::uniform_real_distribution<double> deltaPrior(0.0, 100.0);

  return {pPrior(rng), deltaPrior(rng)};
};

static void bm_sample_one_function(benchmark::State &state) {
  //   state.SetLabel("'pow(., 0.5)' function");
  double epsilon = 1.0;

  int t = 100;
  double pTrue = 0.8, deltaTrue = 5.0;
  Theta thetaTrue = {0.8, 2.0};

  Generator rng(1337);
  Sample obsData = sample_theta(rng, t, thetaTrue);
  SumStats obsSS = get_summary_statistics(obsData);

  int seed = 0;
  for (auto _ : state) {
    sample_one(seed, sample_prior_function, obsSS, t, epsilon);
  }
}

// BENCHMARK(bm_sample_one_function)->Unit(benchmark::kMicrosecond);

// Sample sample_theta_erlang(Generator &rng, int t, const Theta &theta) {

//   double p = theta[0];
//   double delta = theta[1];

//   std::geometric_distribution<int> Geom(1 - p);
//   // std::exponential_distribution<double> Exp(1 / delta);

//   std::vector<double> x(t);
//   for (int s = 0; s < t; s++) {
//     int n_s = Geom(rng);

//     // Assume sum aggregator for now
//     x[s] = 0.0;

//     if (n_s > 0) {
//       std::gamma_distribution<double> Usum(n_s, delta);
//       x[s] += Usum(rng);
//     }
//   }
//   return std::move(x);
// }

// Theta sample_one_erlang(int seed, Theta sample_prior(Generator &),
//                         SumStats obsSS, int t, double epsilon,
//                         int maxAttempts = 1e9) {

//   Generator rng(seed);

//   for (int attempt = 0; attempt < maxAttempts; attempt++) {
//     Theta guess = sample_prior(rng);

//     Sample fakeData = sample_theta_erlang(rng, t, guess);
//     SumStats fakeSS = get_summary_statistics(fakeData);

//     double d = distance(obsSS, fakeSS);
//     if (d < epsilon) {
//       return guess;
//     }
//   }

//   return {};
// }

static void bm_exponential(benchmark::State &state) {

  Generator rng(1337);
  int n = 0;

  Generator rng2(1234);
  std::uniform_real_distribution<double> Unif(0.0, 5.0);

  for (auto _ : state) {
    double delta = Unif(rng2);
    std::exponential_distribution<double> Exp(1 / delta);

    double x_s = 0.0;
    for (int i = 0; i < n; i++) {
      x_s += Exp(rng);
    }
    n = (n + 1) % 100;
  }
}

BENCHMARK(bm_exponential)->Unit(benchmark::kMicrosecond);

static void bm_erlang(benchmark::State &state) {

  Generator rng(1337);
  Generator rng2(1234);
  int n = 0;

  std::uniform_real_distribution<double> Unif(0.0, 5.0);

  for (auto _ : state) {
    if (n > 0) {
      double delta = Unif(rng2);
      std::gamma_distribution<double> Usum(n, delta);
      double x_s = Usum(rng);
    }
    n = (n + 1) % 100;
  }
}

BENCHMARK(bm_erlang)->Unit(benchmark::kMicrosecond);

// static void bm_sample_one_erlang(benchmark::State &state) {
//   //   state.SetLabel("'pow(., 0.5)' function");
//   double epsilon = 1.0;

//   int t = 100;
//   double pTrue = 0.8, deltaTrue = 5.0;
//   Theta thetaTrue = {0.8, 2.0};

//   Generator rng(1337);
//   Sample obsData = sample_theta_erlang(rng, t, thetaTrue);
//   SumStats obsSS = get_summary_statistics(obsData);

//   auto sample_prior = [](Generator &rng) -> Theta {
//     const std::uniform_real_distribution<double> pPrior(0.0, 1.0);
//     const std::uniform_real_distribution<double> deltaPrior(0.0, 100.0);
//     return {pPrior(rng), deltaPrior(rng)};
//   };

//   int seed = 0;
//   for (auto _ : state) {
//     sample_one_erlang(seed, sample_prior, obsSS, t, epsilon);

//   double p = theta[0];
//   double delta = theta[1];

//   std::geometric_distribution<int> Geom(1 - p);
//   // std::exponential_distribution<double> Exp(1 / delta);

//   std::vector<double> x(t);
//   for (int s = 0; s < t; s++) {
//     int n_s = Geom(rng);

//     // Assume sum aggregator for now
//     x[s] = 0.0;

//     if (n_s > 0) {
//       std::gamma_distribution<double> Usum(n_s, delta);
//       x[s] += Usum(rng);
//     }
//   }

//   }
// }

// BENCHMARK(bm_sample_one_erlang)->Unit(benchmark::kMicrosecond);

template <class Generator>
Sample sample_theta_temp(Generator &rng, int t, const Theta &theta) {

  double p = theta[0];
  double delta = theta[1];

  std::geometric_distribution<int> Geom(1 - p);
  std::exponential_distribution<double> Exp(1 / delta);

  std::vector<double> x(t);
  for (int s = 0; s < t; s++) {
    int n_s = Geom(rng);

    // Assume sum aggregator for now
    x[s] = 0.0;
    for (int k = 0; k < n_s; k++) {
      x[s] += Exp(rng);
    }
  }
  return std::move(x);
}

template <class Generator>
Theta sample_one_temp(int seed, Theta sample_prior(Generator &), SumStats obsSS,
                      int t, double epsilon, int maxAttempts = 1e9) {

  Generator rng(seed);

  for (int attempt = 0; attempt < maxAttempts; attempt++) {
    Theta guess = sample_prior(rng);

    Sample fakeData = sample_theta_temp<Generator>(rng, t, guess);
    SumStats fakeSS = get_summary_statistics(fakeData);

    double d = distance(obsSS, fakeSS);
    if (d < epsilon) {
      return guess;
    }
  }

  return {};
}

static void bm_sample_one_temp(benchmark::State &state) {
  double epsilon = 1.0;

  int t = 100;
  double pTrue = 0.8, deltaTrue = 5.0;
  Theta thetaTrue = {0.8, 2.0};

  std::minstd_rand rng(1337);
  Sample obsData = sample_theta_temp(rng, t, thetaTrue);
  SumStats obsSS = get_summary_statistics(obsData);

  auto sample_prior = [](std::minstd_rand &rng) -> Theta {
    std::uniform_real_distribution<double> pPrior(0.0, 1.0);
    std::uniform_real_distribution<double> deltaPrior(0.0, 100.0);
    return {pPrior(rng), deltaPrior(rng)};
  };

  int seed = 0;
  for (auto _ : state) {
    sample_one_temp<std::minstd_rand>(seed, sample_prior, obsSS, t, epsilon);
  }
}

BENCHMARK(bm_sample_one_temp)->Unit(benchmark::kMicrosecond);

static void bm_sample_one_twister(benchmark::State &state) {
  double epsilon = 1.0;

  int t = 100;
  double pTrue = 0.8, deltaTrue = 5.0;
  Theta thetaTrue = {0.8, 2.0};

  std::mt19937 rng(1337);
  Sample obsData = sample_theta_temp(rng, t, thetaTrue);
  SumStats obsSS = get_summary_statistics(obsData);

  auto sample_prior = [](std::mt19937 &rng) -> Theta {
    std::uniform_real_distribution<double> pPrior(0.0, 1.0);
    std::uniform_real_distribution<double> deltaPrior(0.0, 100.0);
    return {pPrior(rng), deltaPrior(rng)};
  };

  int seed = 0;
  for (auto _ : state) {
    sample_one_temp<std::mt19937>(seed, sample_prior, obsSS, t, epsilon);
  }
}

BENCHMARK(bm_sample_one_twister)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();