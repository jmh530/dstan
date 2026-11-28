# DStan

DStan runs Stan inference inside a D process. It can compile a Stan source file using TinyStan, open an existing model library, sample one or more chains with NUTS, and optimize with L-BFGS. Results land in contiguous D memory. The initial release target is TinyStan **0.3.3**, built with Stan **2.39.0**.

```d
import dstan;

auto model = Model(compileModel("normal.stan"), `{"mu": 2.5}`, 123);
SampleOptions settings;
settings.seed = 321;
settings.draws = 4000;  // post-warmup draws per chain
settings.chains = 2;
settings.threads = 2;
auto fit = model.sample(settings);
assert(fit.shape == [2, 4000, 8]);
assert(fit.names[7] == "theta");
double firstTheta = fit[0, 0, 7];
```

Chains and threads both default to one. Chain IDs start at one. Set the desired thread count on the first inference call using a model library. Optimization requests one thread. TinyStan 0.3.3 initializes a persistent Stan thread pool, so a later request can leave the original limit in effect. Use a fresh process when you need a different limit. A threaded build accepts a positive count or `-1` for the available cores.

Set `settings.saveWarmup = true` to include warmup draws. Within each chain, `fit.savedWarmup` is the index of the first posterior row. With 1,000 warmup iterations and the settings above, the shape is `[2, 5000, 8]` and posterior rows start at index 1,000.

Set `settings.initialValues` to an array of in-memory JSON objects containing constrained parameter values. An empty array uses Stan's default initialization, one object applies to every chain, and one object per chain selects distinct starts in chain order. For the two-chain example:

```d
settings.initialValues = [`{"theta":-2}`, `{"theta":2}`];
auto initialized = model.sample(settings);
```

`settings.initRadius` defaults to 2.0. It controls random initialization of unspecified coordinates on the unconstrained scale. Set it to zero to initialize those coordinates at zero.

Set `settings.stepSize` to choose the initial NUTS step size for all chains. It defaults to `1.0` and accepts finite values greater than zero. Stan can adjust the step size during warmup.

Set `settings.metric` to choose the NUTS metric. During warmup, the default `Metric.diagonal` adapts a scale for each unconstrained parameter. `Metric.dense` adapts a full matrix to account for correlations and requires memory quadratic in the number of unconstrained parameters. `Metric.unit` keeps the identity metric and adapts only the step size.

The initial inverse mass matrix defaults to identity. To supply your own, set `settings.initialInverseMetric` to one contiguous block per chain, in chain order. For `p = model.numFreeParams`, each diagonal block contains `p` positive entries and each dense block contains a symmetric positive-definite `p × p` matrix flattened in column-major order. For example, with two chains and two unconstrained parameters, `[0.5, 2.0, 1.5, 0.75]` supplies diagonal entries `[0.5, 2.0]` to the first chain and `[1.5, 0.75]` to the second. Repeat the block for each chain to share a metric. Leave the slice empty for `Metric.unit`.

Set `settings.adaptDelta` to choose the target average acceptance probability (default `0.8`, strictly between zero and one). Set `settings.adapt = false` to keep the configured nominal step size and initial metric fixed. Warmup and `saveWarmup` still apply, and Stan ignores `adaptDelta` in this mode.

Use `sampleWithAdaptation` to keep the final tuning values alongside the draws. It requires adaptation to be enabled and works with warmup draws discarded. For example, to reuse just the learned metric on the same model with the same chain count:

```d
auto fit = model.sampleWithAdaptation(settings);
auto finalStepForFirstChain = fit.stepSizes[0];
settings.initialInverseMetric = fit.inverseMetric;
settings.adapt = false;
auto fixed = model.sample(settings);
```

`fit.draws` has the usual chain, draw, and column layout. `fit.stepSizes` holds one final nominal step size per chain. `fit.inverseMetric` uses the same per-chain layout as `initialInverseMetric` and is empty for the unit metric.

Set `settings.stepSizeJitter` to vary the step size between trajectories. The default is `0.0` (disabled). For example, `0.25` allows up to 25% variation above or below the current nominal step size. DStan accepts finite values in `[0, 1)` because the Stan 2.39.0 sampler bundled with TinyStan 0.3.3 silently ignores exactly `1.0`. [Stan issue #3462](https://github.com/stan-dev/stan/issues/3462) tracks the intended upper bound.

Set `settings.maxTreeDepth` to cap tree growth for each NUTS transition. It accepts integers from `1` through `31` and defaults to `10`. A full depth-31 transition uses `2^31 - 1` leapfrog steps, the largest count Stan 2.39.0 can represent in its signed 32-bit counter. The sampler can stop earlier. The `treedepth__` diagnostic reports the depth reached on each transition.

To reuse storage, allocate `new double[model.requiredCount(settings)]` and pass the slice to `model.sampleInto(buffer, settings)`. The returned count is measured in doubles. To reuse a compiled model, pass the path to its `_model.so` directly to `Model`. Model construction and sampling have separate seeds. The defaults use seed one for each.

## L-BFGS optimization

`model.optimize()` maximizes the model's log density, including any priors in the model. The normal model above has its maximum at `theta = 2.5`.

```d
OptimizeOptions options;
options.seed = 321;
options.initialValues = `{"theta": -2}`;
auto mode = model.optimize(options);
assert(mode.names == ["lp__", "converged__", "theta"]);
double thetaAtMode = mode.values[2];
```

The defaults use seed one, random initialization and at most 2,000 iterations. Set `options.maxIterations` to change the limit. The first two values are the log objective and Stan's numeric termination code. Code 40 means the iteration limit was reached and the row contains the last iterate. The remaining entries are flattened constrained outputs, including transformed parameters and generated quantities.

By default, optimization finds a mode of the density on the constrained parameter scale. Set `options.jacobian = true` to include the log Jacobian of the transformation from unconstrained to constrained parameters. For a positive `theta ~ lognormal(0, 1)`, the default returns `theta` near `exp(-1)`. With the adjustment, the density of `log(theta)` has its mode at zero, so the returned `theta` is near one. Both runs report parameters on their original constrained scale.

## Viewing draws with Mir

The [Mir example](examples/mir-draws/source/columns.d) uses `fit.values.sliced(fit.shape)` to view the existing array as chains, saved draws and columns. Selecting one chain and a named column from `fit.savedWarmup` onward gives a posterior series with a stride of `fit.shape[2]` doubles. Writes through this view update the original draws. Only the example depends on mir-algorithm.

Run the layout test with `dub test --root=examples/mir-draws`. To sample the normal model and print each chain's posterior mean, use the normal example command for your platform below with `--root=examples/mir-draws`.

## Interrupts

During sampling, catch `StanException` and check `category == 2` to recognize a user interrupt. Local integration tests sent SIGINT on Ubuntu/WSL and programmatically generated Ctrl+C and Ctrl+Break events in private native Windows consoles. For each event, single-chain and parallel two-chain runs restored the application's prior handler and allowed another run on the same model. The Python interrupt test driver builds a test model in a temporary directory and runs each case in a child process with a timeout. Use a writable temporary path without spaces, as for other TinyStan builds.

## Build and run on Linux

Use DUB and an ImportC-capable compiler with D frontend 2.112.0 or later and its C preprocessor. DStan imports the TinyStan header directly. Ordinary library builds and unit tests need only that D toolchain.

```sh
dub build
dub test
```

For model compilation install GNU Make, a C++17 toolchain, and the **complete** TinyStan release archive. Extract it to a writable directory. The upstream Makefile can download stanc3 on first use. Use source and build paths without spaces for compilation. The tagged recipes interpolate those paths into shell commands. [TinyStan release archive](https://github.com/WardBrian/tinystan/releases/tag/v0.3.3)

```sh
export TINYSTAN=/absolute/path/to/tinystan-0.3.3
dub run --root=examples/normal -- "$PWD/tests/models/normal.stan"
dub run --root=tests/integration -- "$PWD/tests/models"
python3 tests/interrupt/run.py tests/integration/dstan-integration
```

`CompileOptions.root` takes precedence over `TINYSTAN`. Otherwise DStan looks in `.tinystan/tinystan-0.3.3` under your home directory (the user profile on Windows). Set `CompileOptions.make` or `MAKE` to select GNU Make. With empty `stancArgs`, compilation inherits `STANCFLAGS` from Make, including settings in `make/local`. A nonempty `stancArgs` replaces that setting. DStan supplies the TinyStan root as an include search directory by default.

DStan asks Make to retranslate a model whose source contains `#`, so edits to nested Stan includes take effect on the next compilation. This also rebuilds a model containing `#` in a comment or string. Other models use upstream timestamp checks. Use a fresh build directory or rebuild affected targets after changing flags, compiler, architecture or upstream source. Keep Windows and Linux model outputs in separate directories.

On Linux, reopening a rebuilt library at the same path can still run the previous model, even after `Model.close()`. In a long-running process, build the new revision under a distinct filename and pass that path to `Model`. To reuse the original path, start a fresh process after rebuilding.

## Native Windows

The native x64 integration passed with LDC 1.43.0, portable w64devkit 2.11.0 (MinGW-w64 GCC 16.2.0), and Git for Windows' MSYS shell for Make recipes. The D executable loads the resulting PE DLL with the upstream `_model.so` suffix. Supply explicit DLL directories for TBB and the MinGW runtime. They are independent of the tool PATH used during compilation.

```powershell
$env:TINYSTAN = 'C:/stan/tinystan-0.3.3'
$mingw = 'C:/tools/w64devkit/bin'
$env:PATH = "$mingw;$env:PATH"
$env:MAKEFLAGS = 'SHELL=C:/Progra~1/Git/usr/bin/sh.exe O=1'
$tbb = "$env:TINYSTAN/stan/lib/stan_math/lib/tbb"
dub run --root=examples/normal -- "$PWD/tests/models/normal.stan" $tbb $mingw
dub build --root=tests/integration
python tests/interrupt/run.py tests/integration/dstan-integration.exe $tbb $mingw
```

Adjust the shell path to your Git installation. With the portable toolchain's default shell, the pinned TBB build failed at runtime-version detection. A previously compiled library can be loaded from a path containing spaces. WSL uses the Linux setup and produces an ELF shared object.

## Validation and licensing

The normal-model fixture targets a posterior mean of 2.5 and standard deviation of 1. Integration checks both allocation paths, multiple loaded models, generated quantities, diagnostics, loader errors and resource lifetimes. The optional Python 0.3.3 reference compares the same compiled model and settings. Commands for the ABI fixtures and Python comparison are in the [CI workflow](.github/workflows/test.yml).

Original DStan code is pending maintainer license selection. The two upstream C headers retain TinyStan's [BSD-3-Clause notice](source/dstan/internal/TinyStan-LICENSE.txt). See the [architecture](docs/architecture.md) and [roadmap](docs/roadmap.md) for implementation limits and future work.
