// fully unpooled model
data {
int N;
array[N] int bicycles;
array[N] int total;
}
parameters {
vector<lower=0, upper=1>[N] theta;
}
model {
for (i in 1:N) {
  theta[i] ~ beta(1, 3);
  bicycles[i] ~ binomial(total[i], theta[i]);
}
}
