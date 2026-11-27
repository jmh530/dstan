"""Compare a D integration summary to Python using the same compiled normal model."""
import math
import re
import sys
from pathlib import Path

import numpy as np
import tinystan
from tinystan.model import HMCMetric

library, summary, prefix, mode = sys.argv[1:]
threads = {"serial": 1, "parallel": 2}[mode]
model = tinystan.Model(library)
if mode == "serial":
    fit = model.sample(
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

    structured = tinystan.Model(str(Path(library).with_name("structured_model.so")))
    for metric in (HMCMetric.UNIT, HMCMetric.DIAGONAL, HMCMetric.DENSE):
        base = f"{prefix}.{metric.name.lower()}"
        settings = dict(num_chains=2, seed=321, num_warmup=200, num_samples=20, num_threads=1, metric=metric)
        tuned = structured.sample("{}", save_inv_metric=True, **settings)
        np.testing.assert_allclose(np.fromfile(base + ".step-sizes"), tuned.stepsize, rtol=1e-10, atol=1e-10)
        initial_metric = None
        if metric != HMCMetric.UNIT:
            initial_metric = tuned.inv_metric
            np.testing.assert_allclose(np.fromfile(base + ".inverse-metric"), initial_metric.ravel(), rtol=1e-10, atol=1e-10)
        reused = structured.sample("{}", init_inv_metric=initial_metric, adapt=False, **settings)
        np.testing.assert_allclose(np.fromfile(base + ".reused").reshape(2, 20, 10), reused.data, rtol=1e-10, atol=1e-10)
        for adapt in (False, True):
            reference = structured.sample(
                "{}", num_chains=2, seed=321, num_warmup=200, num_samples=20,
                num_threads=1, save_warmup=True, metric=metric, adapt=adapt,
            )
            suffix = "adapted" if adapt else "fixed"
            draws = np.fromfile(f"{prefix}.{metric.name.lower()}.{suffix}", dtype=np.float64)
            np.testing.assert_allclose(draws.reshape(2, 220, 10), reference.data, rtol=1e-10, atol=1e-10)
            if metric == HMCMetric.UNIT:
                continue
            initial = (np.array([[0.5, 2.0], [1.5, 0.75]]) if metric == HMCMetric.DIAGONAL else
                       np.array([[[0.5, 0.25], [0.25, 2.0]], [[1.5, -0.125], [-0.125, 0.75]]]))
            reference = structured.sample(
                "{}", num_chains=2, seed=321, num_warmup=200, num_samples=20, num_threads=1,
                save_warmup=True, metric=metric, adapt=adapt, init_inv_metric=initial,
            )
            draws = np.fromfile(f"{prefix}.{metric.name.lower()}.{suffix}.initial", dtype=np.float64)
            np.testing.assert_allclose(draws.reshape(2, 220, 10), reference.data, rtol=1e-10, atol=1e-10)

# Compare each chain's complete draw matrix, including all diagnostic columns.
for save_warmup, rows, suffix in [(False, 4000, ""), (True, 5000, ".warmup")]:
    reference = model.sample(
        '{"mu":2.5}', num_chains=2, id=1, seed=321, num_warmup=1000,
        num_samples=4000, num_threads=threads, save_warmup=save_warmup,
    )
    assert reference.data.shape == (2, rows, 8)
    draws = np.fromfile(f"{prefix}.{mode}{suffix}", dtype=np.float64).reshape(reference.data.shape)
    np.testing.assert_allclose(draws, reference.data, rtol=1e-10, atol=1e-10)
for suffix, settings in [
    ("initialized", dict(inits=[{"theta": -2}, {"theta": 2}], init_radius=0.75)),
    ("zero-radius", dict(inits="{}", init_radius=0.0)),
    ("step-size", dict(stepsize=0.25)),
    ("adapt-delta", dict(stepsize=0.25, delta=0.9)),
    ("no-adapt", dict(stepsize=0.25, adapt=False)),
    ("jitter", dict(stepsize=0.25, stepsize_jitter=0.25)),
    ("depth", dict(stepsize=0.25, max_depth=1)),
]:
    reference = model.sample(
        '{"mu":2.5}', num_chains=2, id=1, seed=321, num_warmup=100,
        num_samples=20, num_threads=threads, save_warmup=True,
        **settings,
    )
    draws = np.fromfile(f"{prefix}.{mode}.{suffix}", dtype=np.float64).reshape(2, 120, 8)
    np.testing.assert_allclose(draws, reference.data, rtol=1e-10, atol=1e-10)
print(f"{mode}: both chains match Python for all tested NUTS settings")
