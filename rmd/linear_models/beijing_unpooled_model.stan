// Fully unpooled model for the Beijing air quality data, modeling log ozone as a function of temp
// Essentially, 12 independent regressions here

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  int<lower=0> p;
  matrix[N, p] temp;
  matrix[N, p] log_ozone;
}

parameters {
  vector[p] alpha;
  vector[p] beta;
  real<lower=0> sigma;
}

model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 0.1);
  sigma ~ exponential(1);
  for (i in 1:p) {
    col(log_ozone, i) ~ normal(col(temp, i) * beta[i] + alpha[i], sigma);
  }
}