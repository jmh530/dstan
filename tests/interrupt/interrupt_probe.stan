functions {
  int interrupt_workers(int workers);
}
data {
  int<lower=1> workers;
}
parameters {
  real theta;
}
model {
  theta ~ std_normal();
}
generated quantities {
  int marker = interrupt_workers(workers);
}
