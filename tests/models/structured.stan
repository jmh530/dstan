parameters {
  vector[2] x;
}
model {
  x ~ std_normal();
}
generated quantities {
  real total = sum(x);
  matrix[2, 3] grid;
  array[2] vector[3] points;
  for (i in 1:2) {
    for (j in 1:3) {
      grid[i, j] = 10 * i + j;
      points[i, j] = 100 * i + j;
    }
  }
}
