data {
  int N;
  vector[N] temp;
  vector[N] windspeed;
  array[N] int cnt;
}

parameters {
  real alpha;
  real beta_temp;
  real beta_wind;
}

model {
  alpha ~ normal(0, 5);
  beta_temp ~ normal(0, 2);
  beta_wind ~ normal(0, 2);
  vector[N] theta = exp(alpha + beta_temp * temp + beta_wind * windspeed);
  cnt ~ poisson(theta);
}

generated quantities {
  array[N] int bikes_pred;
  bikes_pred = poisson_rng(
      exp(alpha + beta_temp * temp + beta_wind * windspeed)
    );
}