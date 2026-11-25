parameters {
  vector[2] x;
}
model {
  x ~ std_normal();
}
generated quantities {
  real total = sum(x);
}
