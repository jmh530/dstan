parameters {
  real<lower=0> theta;
}
model {
  theta ~ lognormal(0, 1);
}
