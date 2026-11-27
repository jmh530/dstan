functions {
  int interrupt_workers(int workers, int ctrl_break);
}
data {
  int<lower=1> workers;
  int<lower=0, upper=1> ctrl_break;
}
parameters {
  real theta;
}
model {
  theta ~ std_normal();
}
generated quantities {
  int marker = interrupt_workers(workers, ctrl_break);
}
