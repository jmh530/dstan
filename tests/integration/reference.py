"""Compare a D integration summary to Python using the same compiled normal model."""
import math
import re
import sys
from pathlib import Path

import tinystan

library, summary = sys.argv[1:]
fit = tinystan.Model(library).sample(
    '{"mu":2.5}', num_chains=1, seed=321, num_warmup=1000,
    num_samples=4000, num_threads=1,
)
assert fit.data.shape == (1, 4000, 8)
assert fit.raw_parameters == [
    "lp__", "accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__",
    "divergent__", "energy__", "theta",
]
theta = fit["theta"].reshape(-1)
match = re.search(r"shape=\[1, 4000, 8\] mean=([\d.]+) sd=([\d.]+)", Path(summary).read_text())
assert match, "D summary missing"
for reference, observed in zip((theta.mean(), theta.std(ddof=1)), match.groups()):
    assert math.isclose(reference, float(observed), abs_tol=1e-10)
print(f"Python theta mean={theta.mean():.12f} sd={theta.std(ddof=1):.12f}")
