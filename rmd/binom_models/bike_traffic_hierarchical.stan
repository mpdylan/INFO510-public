// hierarchical model
data {
int N;
array[N] int bicycles;
array[N] int total;
}
parameters {
real <lower = 0, upper = 1>mu;
real <lower = 0> eta;
vector<lower=0, upper=1>[N] theta;
}
transformed parameters {
real a = eta * mu;
real b = eta * (1 - mu);
}
model {
mu ~ beta(1, 3);
eta ~ exponential(1);
for (i in 1:N) {
  theta[i] ~ beta(a, b);
  bicycles[i] ~ binomial(total[i], theta[i]);
}
}