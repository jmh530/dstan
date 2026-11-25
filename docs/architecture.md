# Architecture

## Build and load

`import dstan` exports compilation, model and result modules. Ordinary DUB builds import declarations from a small C translation unit and the two public headers copied verbatim from TinyStan v0.3.3. The private loader derives function-pointer types with `typeof(&function)`. This compiled on Windows DMD and LDC and Ubuntu LDC. The package needs no separately generated bindings, import library, C++ shim or DUB dependencies. The integration workflow compares both headers with the release.

Each model library has its own addresses and allocator. The loader first checks the exact TinyStan API version, then resolves the milestone's required symbols. It reports the embedded Stan version separately. POSIX uses local, eager symbol resolution. Windows uses a wide absolute path, scoped dependency-directory registrations and per-handle symbol lookup. A short lock serializes DStan loader registrations. Dependencies must remain available to the OS loader for the model's lifetime. A rebuilt binary should be loaded after closing the previous owner.

Model compilation runs the upstream Makefile in its own directory and preserves combined standard output and error, status and arguments on failure. Runtime model loading works independently of source discovery. Use trusted model binaries, sources and build flags. A loaded model executes native code in the calling process.

## Ownership and threading

Keep `Model` as a local, thread-confined value. It owns its module and foreign handle, disables copying, and destroys the model before releasing the module reference. Its destructor also handles ordinary exception unwinding. Explicit `close` is idempotent. Avoid GC-allocated owners when deterministic cleanup matters. Finish active calls before closing or replacing a model. Output names and error messages are copied while their owners are live. A `StanException` retains the upstream category. Each error object is freed through its originating library.

The synchronous sampler supplies a null print callback, one chain, one thread and chain ID one. It serializes DStan sampling calls because upstream installs process-wide interrupt handlers. Applications embedding other Stan interfaces must coordinate their calls independently. The model can consume transformed-data randomness during construction, before sampling begins. JSON object text is passed as a NUL-terminated string and parsed upstream.

## Output

The flat index is `(chain * savedDraws + draw) * columns + column`. All seven diagnostic columns precede model outputs. TinyStan reports flattened constrained output names in output order, including transformed parameters and generated quantities. The number of unconstrained parameters cannot size this buffer. DStan checks capacity and the signed-int per-chain limit in upstream 0.3.3, and checks byte-count overflow before allocating.

`sampleInto` borrows exclusive, aligned storage for one call. Its private trusted bridge registers the address of a local storage pointer as a temporary GC scan range and removes it on exit. The upstream sampler writes synchronously and releases its writers before returning. An error can leave a partially written prefix. `sample` keeps a D-owned slice and copied names usable after model closure. TinyStan copies each Stan draw vector into that storage. The interface avoids result CSV, parsing and a subsequent full-buffer copy.

A native Windows experiment constructed `fit.values.sliced(fit.shape)` with mir-algorithm, checked the address of an interior element, closed the model and read the view after GC collection with its D storage owner live. A fixed parameter within one chain has stride `columns`. Applications using Mir can build this view today and retain the backing storage. No Mir dependency is required by the core.

## Validation record

Ubuntu/WSL x86-64 used GCC 13.3, GNU Make 4.3 and LDC 1.43.0. Windows x64 used LDC 1.43.0 with a MinGW-built model. Both returned shape `[1, 4000, 8]` and theta mean 2.476691555006 and sample standard deviation 1.009324489816. Python TinyStan 0.3.3 agreed on the same Linux binary. The integration acceptance bounds are statistical; these observed values are specific to the tested configurations. Windows DMD 2.114.0 also passed unit tests and the real-model integration suite using those same native model binaries. The C probe exercises the callback ABI and rejects incompatible and incomplete libraries. Hosted CI, macOS and GDC require separate validation.
