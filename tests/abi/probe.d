import dstan;
import std.exception : enforce;
import std.string : indexOf;
import dstan.internal.tinystan;

private bool observed;
private extern(C) void callback(const(char)* message, size_t length, bool isError) nothrow
{
    observed = length == 2 && message[0 .. 2] == "ok" && isError;
}

private void withSymbol(T)(string path, string symbol, scope void delegate(T) use)
{
    version (Windows)
    {
        import core.sys.windows.windows;
        import std.string : toStringz;
        import std.utf : toUTF16z;
        auto lib = LoadLibraryW(path.toUTF16z);
        enforce(lib);
        scope(exit) FreeLibrary(lib);
        auto address = GetProcAddress(lib, symbol.toStringz);
    }
    else
    {
        import core.sys.posix.dlfcn;
        import std.string : toStringz;
        auto lib = dlopen(path.toStringz, RTLD_NOW | RTLD_LOCAL);
        enforce(lib);
        scope(exit) dlclose(lib);
        auto address = dlsym(lib, symbol.toStringz);
    }
    enforce(address);
    use(cast(T) address);
}

private struct InferenceProbe
{
    import core.sync.semaphore : Semaphore;
    Semaphore sampleEntered, contenderStarting, contenderEntered;
    bool overlapped;
}

private extern(C) int inferenceEntry(void* context, bool contender) nothrow
{
    import core.time : msecs, seconds;
    auto probe = cast(InferenceProbe*) context;
    try
    {
        if (contender) probe.contenderEntered.notify();
        else
        {
            probe.sampleEntered.notify();
            if (!probe.contenderStarting.wait(10.seconds)) return 1;
            // Keep the native sampler active while the other thread attempts inference.
            probe.overlapped = probe.contenderEntered.wait(500.msecs);
        }
        return 0;
    }
    catch (Throwable) { return 1; } // Never unwind through the C fixture.
}

private void checkInferenceLock(string path, string operation, scope void delegate() compete)
{
    import core.sync.semaphore : Semaphore;
    import core.thread : Thread;
    import core.time : msecs, seconds;

    alias Hook = typeof(&inferenceEntry);
    alias SetHook = extern(C) void function(Hook, void*);
    withSymbol!SetHook(path, "dstan_set_inference_probe", (setHook) {
        InferenceProbe probe = {new Semaphore, new Semaphore, new Semaphore};
        setHook(&inferenceEntry, &probe);
        scope(exit) setHook(null, null);
        auto sampler = new Thread({
            auto model = Model(path);
            SampleOptions options;
            options.draws = 1;
            model.sample(options);
        });
        {
            sampler.start();
            scope(exit) sampler.join();
            enforce(probe.sampleEntered.wait(10.seconds), "Sampler did not enter the ABI fixture");
            probe.contenderStarting.notify();
            compete();
        }
        enforce(!probe.overlapped, operation ~ " entered the ABI fixture while sampling was active");
        enforce(probe.contenderEntered.wait(0.msecs), operation ~ " did not enter the ABI fixture");
    });
}

private void checkSampling(string[] paths)
{
    auto model = Model(paths[0]);
    auto mode = model.optimize();
    enforce(mode.names == ["lp__", "converged__", "adapt", "delta", "metric"]);
    enforce(mode.values == [0, 30, 1, 2000, 0], "Wrong C ABI optimizer defaults");
    OptimizeOptions optimization;
    optimization.seed = 123;
    optimization.maxIterations = 17;
    optimization.initialValues = `{"adapt":1}`;
    optimization.jacobian = true;
    auto adjusted = model.optimize(optimization);
    enforce(adjusted.values == [1, 30, 123, 17, 1], "Wrong explicit optimizer settings");
    auto approx = model.laplaceSample(adjusted, 2, 42);
    enforce(approx.shape == [1, 2, 5] && approx.savedWarmup == 0);
    enforce(approx.names == ["log_p__", "log_q__", "adapt", "delta", "metric"]);
    enforce(approx.values == [42, 1, 123, 17, 1, 42, 1, 123, 17, 1]);
    auto defaultApprox = model.laplaceSample(mode);
    enforce(defaultApprox.shape == [1, 1000, 5] && defaultApprox.values[0 .. 2] == [1, 0]);
    version (Windows)
    {
        import core.sys.windows.windows;
        import std.utf : toUTF16z;
        // The model has already bound to its adjacent tbb.dll. Introduce a
        // second DLL with the same basename before the first multi-chain call.
        auto bound = GetModuleHandleW(paths[1].toUTF16z);
        enforce(bound, "Model runtime fixture missing");
        auto other = LoadLibraryW(paths[2].toUTF16z);
        enforce(other && other != bound, "Expected two distinct tbb.dll modules");
    }

    SampleOptions options;
    options.draws = 1;
    options.chains = 2;
    double[1] backing;
    options.initialInverseMetric = backing[0 .. 0];
    auto defaults = model.sample(options);
    enforce(defaults[0, 0, 9] == diagonal, "Expected diagonal metric by default");
    foreach (i, metric; [Metric.unit, Metric.diagonal, Metric.dense])
    {
        options.metric = metric;
        enforce(model.sample(options)[0, 0, 9] == [unit, diagonal, dense][i], "Wrong C ABI metric");
    }
    import std.exception : assertThrown, collectExceptionMsg;
    auto tooLarge = options;
    tooLarge.chains = int.max;
    tooLarge.initialInverseMetric = [1.0];
    auto message = collectExceptionMsg(model.requiredCount(tooLarge));
    enforce(message == "Inverse metric dimensions exceed supported limits",
        "Unexpected overflow diagnostic: " ~ message);
    static if (size_t.sizeof == 8)
    {
        tooLarge.initialInverseMetric = null;
        enforce(collectExceptionMsg(model.sampleWithAdaptation(tooLarge)) == message,
            "Expected metric output overflow before allocation");
    }
    auto missingState = options;
    missingState.metric = Metric.unit;
    enforce(collectExceptionMsg(model.sampleWithAdaptation(missingState)) ==
        "Stan did not return a valid final step size for every chain",
        "Expected the unwritten second-chain step size to be rejected");
    auto oversized = Model(paths[0], `{"oversized":true}`);
    assertThrown(oversized.requiredCount(options)); // Dense at 46,341 free parameters.
    options.metric = cast(Metric) 99;
    assertThrown(model.requiredCount(options));
    options.metric = Metric.diagonal;
    enforce(oversized.requiredCount(options) == defaults.values.length);
    oversized.close();
    enforce(defaults[0, 0, 0] == 2, "Wrong C ABI chain count");
    enforce(defaults[0, 0, 2] == 2 && defaults[0, 0, 3] == 0, "Wrong default initialization");
    enforce(defaults[0, 0, 4] == 1.0, "Wrong default C ABI step size");
    enforce(defaults[0, 0, 5] == 0.0, "Wrong default C ABI step-size jitter");
    enforce(defaults[0, 0, 6] == 10, "Wrong default C ABI maximum tree depth");
    enforce(defaults[0, 0, 7] == 1 && defaults[0, 0, 8] == 0.8, "Wrong default C ABI adaptation settings");
    options.adaptDelta = 0.9;
    enforce(model.sample(options)[0, 0, 8] == 0.9, "Wrong explicit C ABI adaptation target");
    options.adapt = false;
    enforce(model.sample(options)[0, 0, 7] == 0, "Expected adaptation disabled at the C ABI");
    foreach (depth; [1, 31])
    {
        options.maxTreeDepth = depth;
        enforce(model.sample(options)[0, 0, 6] == depth, "Wrong explicit C ABI maximum tree depth");
    }
    import std.math : nextDown;
    foreach (jitter; [0.25, nextDown(1.0)])
    {
        options.stepSizeJitter = jitter;
        enforce(model.sample(options)[0, 0, 5] == jitter, "Wrong explicit C ABI step-size jitter");
    }
    options.stepSizeJitter = 0;
    options.stepSize = 0.25;
    options.initialValues = [`{"theta":-2}`];
    options.initRadius = 0;
    auto sharedInit = model.sample(options);
    enforce(sharedInit[0, 0, 4] == 0.25, "Wrong explicit C ABI step size");
    enforce(sharedInit[0, 0, 2] == 0 && sharedInit[0, 0, 3] == 1, "Wrong shared initialization");
    options.initialValues = [`{"theta":-2}`, `{"theta":2}`];
    options.initRadius = 0.75;
    auto separate = model.sample(options);
    enforce(separate[0, 0, 2] == 0.75 && separate[0, 0, 3] == 2, "Wrong per-chain initialization");
    options.initialValues = null;
    options.chains = 1;
    foreach (requested; [1, 2, -1])
    {
        options.threads = requested;
        enforce(model.sample(options)[0, 0, 1] == requested, "Wrong C ABI thread count");
    }
    model.close();
    version (Windows)
    {
        FreeLibrary(other);
        enforce(GetModuleHandleW(paths[1].toUTF16z) == bound, "Model runtime was unloaded");
        enforce(!GetModuleHandleW(paths[2].toUTF16z), "Unrelated runtime was pinned");
    }
}

void main(string[] args)
{
    version (Windows) enum expectedArgs = 6;
    else enum expectedArgs = 4;
    enforce(args.length == expectedArgs, "Pass the fixture library paths from the CI workflow");
    alias Check = extern(C) bool function(TinyStanMetric, size_t, uint, TINYSTAN_PRINT_CALLBACK);
    withSymbol!Check(args[1], "dstan_check_callback", (check) {
        assert(check(diagonal, size_t.max, uint.max, &callback));
        assert(observed);
    });
    foreach (i, expected; ["Expected TinyStan 0.3.3", "tinystan_stan_version"])
    {
        try
        {
            auto model = Model(args[i + 1]);
            assert(0, "Partial model library was accepted");
        }
        catch (Exception e) { assert(e.msg.indexOf(expected) >= 0, e.msg); }
    }
    checkSampling(args[3 .. $]);
    auto contender = Model(args[3], `{"oversized":true}`); // Distinct native fixture handle.
    auto mode = contender.optimize(); // Prepare the Laplace input before installing the hook.
    checkInferenceLock(args[3], "Optimizer", { contender.optimize(); });
    checkInferenceLock(args[3], "Laplace", { contender.laplaceSample(mode, 1); });
}
