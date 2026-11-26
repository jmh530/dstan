# DStan

DStan runs Stan inference inside a D process. It can compile a Stan source file using TinyStan, open an existing model library, and sample one or more chains with adaptive diagonal-metric NUTS. Draws land in contiguous D memory. The initial release target is TinyStan **0.3.3**, built with Stan **2.39.0**.

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

Chains and threads both default to one. Chain IDs start at one. Set the desired thread count on the first sampling call using a model library. TinyStan 0.3.3 initializes a persistent Stan thread pool, so a later request can leave the original limit in effect. Use a fresh process when you need a different limit. A threaded build accepts a positive count or `-1` for the available cores.

Set `settings.saveWarmup = true` to include warmup draws. Within each chain, `fit.savedWarmup` is the index of the first posterior row. With 1,000 warmup iterations and the settings above, the shape is `[2, 5000, 8]` and posterior rows start at index 1,000.

Set `settings.initialValues` to an array of in-memory JSON objects containing constrained parameter values. An empty array uses Stan's default initialization, one object applies to every chain, and one object per chain selects distinct starts in chain order. For the two-chain example:

```d
settings.initialValues = [`{"theta":-2}`, `{"theta":2}`];
auto initialized = model.sample(settings);
```

`settings.initRadius` defaults to 2.0. It controls random initialization of unspecified coordinates on the unconstrained scale. Set it to zero to initialize those coordinates at zero.

Set `settings.stepSize` to choose the initial NUTS step size for all chains. It defaults to `1.0` and accepts finite values greater than zero. Stan can adjust the step size during warmup.

Set `settings.stepSizeJitter` to vary the step size between trajectories. The default is `0.0` (disabled). For example, `0.25` allows up to 25% variation above or below the current nominal step size. DStan accepts finite values in `[0, 1)` because the Stan 2.39.0 sampler bundled with TinyStan 0.3.3 silently ignores exactly `1.0`. [Stan issue #3462](https://github.com/stan-dev/stan/issues/3462) tracks the intended upper bound.

To reuse storage, allocate `new double[model.requiredCount(settings)]` and pass the slice to `model.sampleInto(buffer, settings)`. The returned count is measured in doubles. To reuse a compiled model, pass the path to its `_model.so` directly to `Model`. Model construction and sampling have separate seeds. The defaults use seed one for each.

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
```

Adjust the shell path to your Git installation. With the portable toolchain's default shell, the pinned TBB build failed at runtime-version detection. A previously compiled library can be loaded from a path containing spaces. WSL uses the Linux setup and produces an ELF shared object.

## Validation and licensing

The normal-model fixture targets a posterior mean of 2.5 and standard deviation of 1. Integration checks both allocation paths, multiple loaded models, generated quantities, diagnostics, loader errors and resource lifetimes. The optional Python 0.3.3 reference compares the same compiled model and settings. Commands for the ABI fixtures and Python comparison are in the [CI workflow](.github/workflows/test.yml).

Original DStan code is pending maintainer license selection. The two upstream C headers retain TinyStan's [BSD-3-Clause notice](source/dstan/internal/TinyStan-LICENSE.txt). See the [architecture](docs/architecture.md) and [roadmap](docs/roadmap.md) for implementation limits and future work.
