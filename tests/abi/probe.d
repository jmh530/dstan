import dstan;
import std.exception : enforce;
import std.string : indexOf;
import dstan.internal.tinystan;

private bool observed;
private extern(C) void callback(const(char)* message, size_t length, bool isError) nothrow
{
    observed = length == 2 && message[0 .. 2] == "ok" && isError;
}

private void checkCallback(string path)
{
    version (Windows)
    {
        import core.sys.windows.windows;
        import std.utf : toUTF16z;
        auto lib = LoadLibraryW(path.toUTF16z);
        enforce(lib);
        scope(exit) FreeLibrary(lib);
        auto address = GetProcAddress(lib, "dstan_check_callback");
    }
    else
    {
        import core.sys.posix.dlfcn;
        import std.string : toStringz;
        auto lib = dlopen(path.toStringz, RTLD_NOW | RTLD_LOCAL);
        enforce(lib);
        scope(exit) dlclose(lib);
        auto address = dlsym(lib, "dstan_check_callback");
    }
    alias Check = extern(C) bool function(TinyStanMetric, size_t, uint, TINYSTAN_PRINT_CALLBACK);
    enforce(address);
    assert((cast(Check)address)(diagonal, size_t.max, uint.max, &callback));
    assert(observed);
}

private void checkSampling(string[] paths)
{
    auto model = Model(paths[0]);
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
    auto defaults = model.sample(options);
    enforce(defaults[0, 0, 0] == 2, "Wrong C ABI chain count");
    enforce(defaults[0, 0, 2] == 2 && defaults[0, 0, 3] == 0, "Wrong default initialization");
    enforce(defaults[0, 0, 4] == 1.0, "Wrong default C ABI step size");
    enforce(defaults[0, 0, 5] == 0.0, "Wrong default C ABI step-size jitter");
    enforce(defaults[0, 0, 6] == 10, "Wrong default C ABI maximum tree depth");
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
    checkCallback(args[1]);
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
}
