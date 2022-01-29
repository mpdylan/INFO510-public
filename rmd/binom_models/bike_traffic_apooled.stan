// fully unpooled model
data {
int N;
vector[N] bicycles;
vector[N] total;
}
parameters {
vector<lower=0, upper=1>[N] theta;
}
model{
for (i in 1:N) {
  theta[i] ~ beta(1, 3);
}
for (i in 1:N) {
  bicycles[i] ~ binomial(total[i], theta[i]);
}
}
