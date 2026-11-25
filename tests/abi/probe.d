import dstan;
import std.exception : enforce;
import std.string : indexOf;
import dstan_abi;

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

void main(string[] args)
{
    enforce(args.length == 3, "Pass the wrong-version and missing-export fixture libraries");
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
}
