# DStan

DStan runs Stan inference inside a D process. It can compile a Stan source file using TinyStan, open an existing model library, and sample one chain with adaptive diagonal-metric NUTS. Draws land in contiguous D memory. The initial release target is TinyStan **0.3.3**, built with Stan **2.39.0**.

```d
import dstan;

auto model = Model(compileModel("normal.stan"), `{"mu": 2.5}`, 123);
SampleOptions settings;
settings.seed = 321;
settings.draws = 4000;
auto fit = model.sample(settings);
assert(fit.names[7] == "theta");
double firstTheta = fit[0, 0, 7];
```

To reuse storage, allocate `new double[model.requiredCount(settings)]` and pass the slice to `model.sampleInto(buffer, settings)`. The returned count is measured in doubles. To reuse a compiled model, pass the path to its `_model.so` directly to `Model`. Model construction and sampling have separate seeds. The defaults use seed one for each.

## Build and run on Linux

Use DUB, a D compiler with ImportC, and its C preprocessor. Ordinary library builds and unit tests need only that D toolchain.

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
