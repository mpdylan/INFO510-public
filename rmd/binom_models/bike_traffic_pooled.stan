// fully pooled model
data {
int bicycles;
int total;
}
parameters {
real<lower=0, upper=1> theta;
}
model {
theta ~ beta(1, 3);
bicycles ~ binomial(total, theta);
}