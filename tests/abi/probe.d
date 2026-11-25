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
    enforce(model.sample(options)[0, 0, 0] == 2, "Wrong C ABI chain count");
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
