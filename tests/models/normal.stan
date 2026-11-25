data {
  real mu;
}
parameters {
  real theta;
}
model {
  theta ~ normal(mu, 1);
}
