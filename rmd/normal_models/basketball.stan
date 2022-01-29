data {
int N;
vector[N] score;
}
parameters {
real mu;
real<lower=0> sigma;
}
model {
mu ~ normal(150, 25);
sigma ~ exponential(1);
score ~ normal(mu, sigma);
}