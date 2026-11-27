# Architecture

## Build and load

`import dstan` exports compilation, model and result modules. DStan imports `dstan.internal.tinystan` directly from TinyStan's `tinystan.h`, which includes `tinystan_types.h`. Both headers are copied verbatim from TinyStan v0.3.3. The private loader derives function-pointer types with `typeof(&function)`. The package needs no separately generated bindings, import library, C++ shim or DUB dependencies. The integration workflow compares both headers with the release.

Each model library has its own addresses and allocator. The loader first checks the exact TinyStan API version, then resolves the milestone's required symbols. It reports the embedded Stan version separately. POSIX uses local, eager symbol resolution. Windows uses a wide absolute path, scoped dependency-directory registrations and per-handle symbol lookup. A short lock serializes DStan loader registrations. Dependencies must remain available to the OS loader for the model's lifetime.

`Model.close()` destroys the Stan model and releases DStan's reference to the shared library. The OS loader can keep the library mapped. On Ubuntu, closing a model, replacing its binary at the same path and reopening it returned the original parameter names. Loading the replacement from a distinct path returned the new names. Use a distinct library file for each revision loaded in one process, or reuse the path in a fresh process.

Model compilation runs the upstream Makefile in its own directory and preserves combined standard output and error, status and arguments on failure. Runtime model loading works independently of source discovery. Use trusted model binaries, sources and build flags. A loaded model executes native code in the calling process.

## Ownership and threading

Keep `Model` as a local, thread-confined value. It owns its module and foreign handle, disables copying, and destroys the model before releasing the module reference. Its destructor also handles ordinary exception unwinding. Explicit `close` is idempotent. Avoid GC-allocated owners when deterministic cleanup matters. Finish active calls before closing or replacing a model. Output names and error messages are copied while their owners are live. A `StanException` retains the upstream category. Each error object is freed through its originating library.

The synchronous sampler passes all chains to TinyStan in one call with a null print callback and starting chain ID one. Both chain count and thread count default to one. It serializes DStan sampling calls because upstream installs process-wide interrupt handlers. Applications embedding other Stan interfaces must coordinate their calls independently. The model can consume transformed-data randomness during construction, before sampling begins. JSON object text is passed as a NUL-terminated string and parsed upstream. Sampling passes a null initialization pointer by default. For a shared start it passes one object directly. For distinct starts it joins the objects in chain order using the separator reported by the loaded model library. DStan rejects embedded NUL and literal separator bytes in initial values. The D-owned input stays rooted until sampling returns.

Stan Math initializes static TBB controls on first use. In fresh Windows and Ubuntu processes, requesting one thread then two yielded active limits of one and one. Reversing the requests yielded two and two. Other active TBB controls can further limit parallelism. On Windows, multi-chain or threaded sampling uses a resolved import address from the model to retain its TBB runtime DLL until process exit so its workers can finish after model closure. Model handles and model-library references are still released by the owner.

## Output

Metric sizing uses the unconstrained parameter count reported by the loaded model. TinyStan 0.3.3 squares that count in a signed 32-bit integer for a dense metric, even with null metric input and output buffers. DStan rejects dense sampling above 46,340 unconstrained parameters before allocating draws or entering the sampler. Available memory can limit dense sampling at much smaller sizes.

The initial inverse metric borrows one contiguous input block per chain. DStan validates the exact length, alignment, finite entries and separation from the writable draw prefix, then roots the input for the synchronous call. Empty input passes a null pointer. TinyStan's per-chain readers copy the values into Stan during initialization. Stan validates diagonal positivity and dense symmetry and positive definiteness.

The flat index is `(chain * savedDraws + draw) * columns + column`. With saved warmup, `savedDraws` is the sum of warmup and post-warmup iterations per chain. `Draws.savedWarmup` records the first posterior row index in each chain. All seven diagnostic columns precede model outputs. TinyStan reports flattened constrained output names in output order, including transformed parameters and generated quantities. The number of unconstrained parameters cannot size this buffer. DStan checks capacity and the signed-int per-chain limit in upstream 0.3.3, and checks the total byte count across all chains before either allocation or a caller-buffer write.

`sampleInto` borrows exclusive, aligned storage for one call. Its private trusted bridge registers the local input and output pointers as a temporary GC scan range and removes it on exit. The upstream sampler writes synchronously and releases its writers before returning. An error can leave a partially written prefix. `sample` keeps a D-owned slice and copied names usable after model closure. TinyStan copies each Stan draw vector into that storage. The interface avoids result CSV, parsing and a subsequent full-buffer copy.

A native Windows experiment constructed `fit.values.sliced(fit.shape)` with mir-algorithm, checked the address of an interior element, closed the model and read the view after GC collection with its D storage owner live. A fixed parameter within one chain has stride `columns`. Applications using Mir can build this view today and retain the backing storage. No Mir dependency is required by the core.

## Validation record

Ubuntu/WSL x86-64 used GCC 13.3, GNU Make 4.3 and LDC 1.43.0. Windows x64 used LDC 1.43.0 with a MinGW-built model. Both returned shape `[1, 4000, 8]` and theta mean 2.476691555006 and sample standard deviation 1.009324489816. Python TinyStan 0.3.3 agreed on the same Linux binary. The integration acceptance bounds are statistical; these observed values are specific to the tested configurations. Windows DMD 2.114.0 also passed unit tests and the real-model integration suite using those same native model binaries. The C probe exercises the callback ABI and rejects incompatible and incomplete libraries. Hosted CI, macOS and GDC require separate validation.
