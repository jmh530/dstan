module dstan.internal.library;

import dstan.internal.tinystan;
import std.path : absolutePath;
import std.string : fromStringz, toStringz;
import std.format : format;

version (Windows)
{
    import core.sys.windows.windows;
    import std.utf : toUTF16z;
    extern(Windows) private void* AddDllDirectory(const(wchar)*);
    extern(Windows) private BOOL RemoveDllDirectory(void*);
}
else version (Posix)
{
    import core.sys.posix.dlfcn;
}
else static assert(false, "Unsupported dynamic loader");

package(dstan) struct Library
{
    @disable this(this);
    private void* handle;
    int[3] stanVersion;
    alias Fn(alias f) = typeof(&f);
    Fn!tinystan_create_model create;
    Fn!tinystan_destroy_model destroy;
    Fn!tinystan_model_param_names names;
    Fn!tinystan_sample sample;
    Fn!tinystan_get_error_type errorType;
    Fn!tinystan_get_error_message message;
    Fn!tinystan_destroy_error freeError;

    private T resolve(T)(string name)
    {
        version (Windows)
            auto address = GetProcAddress(cast(HMODULE) handle, name.toStringz);
        else
            auto address = dlsym(handle, name.toStringz);
        if (!address) throw new Exception("Missing TinyStan symbol " ~ name);
        return cast(T) address;
    }

    void open(string path, string[] dependencyDirs)
    {
        path = absolutePath(path);
        if (path.hasNul) throw new Exception("NUL in model library path");
        version (Windows)
        {
            void*[] cookies;
            scope(exit) foreach_reverse (c; cookies) RemoveDllDirectory(c);
            foreach (dir; dependencyDirs)
            {
                if (dir.hasNul) throw new Exception("NUL in DLL directory");
                auto c = AddDllDirectory(absolutePath(dir).toUTF16z);
                if (!c) throw new Exception(format("DLL directory %s (Windows error %s)", dir, GetLastError()));
                cookies ~= c;
            }
            // winbase.h values; druntime exposes them conditionally on its SDK target.
            enum dllLoadDir = 0x100, defaultDirs = 0x1000;
            handle = LoadLibraryExW(path.toUTF16z, null, dllLoadDir | defaultDirs);
            if (!handle) throw new Exception(format("Loading %s (Windows error %s)", path, GetLastError()));
        }
        else
        {
            if (dependencyDirs.length) throw new Exception("dependencyDirs is a Windows loader option");
            handle = dlopen(path.toStringz, RTLD_NOW | RTLD_LOCAL);
            if (!handle) throw new Exception("Loading " ~ path ~ ": " ~ dlerror().fromStringz.idup);
        }
        scope(failure) close();
        int[3] api;
        auto versionFn = resolve!(Fn!tinystan_api_version)("tinystan_api_version");
        versionFn(&api[0], &api[1], &api[2]);
        if (api != [0, 3, 3]) throw new Exception(format("Expected TinyStan 0.3.3, got %d.%d.%d", api[0], api[1], api[2]));
        versionFn = resolve!(Fn!tinystan_stan_version)("tinystan_stan_version");
        versionFn(&stanVersion[0], &stanVersion[1], &stanVersion[2]);
        create = resolve!(typeof(create))("tinystan_create_model");
        destroy = resolve!(typeof(destroy))("tinystan_destroy_model");
        names = resolve!(typeof(names))("tinystan_model_param_names");
        sample = resolve!(typeof(sample))("tinystan_sample");
        errorType = resolve!(typeof(errorType))("tinystan_get_error_type");
        message = resolve!(typeof(message))("tinystan_get_error_message");
        freeError = resolve!(typeof(freeError))("tinystan_destroy_error");
    }

    void close() nothrow
    {
        if (!handle) return;
        version (Windows) FreeLibrary(cast(HMODULE) handle);
        else dlclose(handle);
        handle = null;
    }
    ~this() { close(); }
}

package(dstan) bool hasNul(string s) pure @safe nothrow @nogc
{
    foreach (c; s) if (!c) return true;
    return false;
}
