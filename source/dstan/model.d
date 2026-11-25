/// Ownership of initialized Stan models.
module dstan.model;

import dstan.internal.library;
import dstan_abi;
import std.string : fromStringz, strip, toStringz;

/// TinyStan status translated into D-owned exception data.
class StanException : Exception
{
    int category; /// TinyStan error type (generic=0, configuration=1, interrupt=2).
    package this(string msg, int category) { super(msg); this.category = category; }
}

/// A scoped, noncopyable owner of one initialized Stan model and its library.
struct Model
{
    @disable this(this);
    private Library lib;
    private TinyStanModel* handle;

    /**
    Params:
        library = Path of a compiled TinyStan 0.3.3 model.
        json = In-memory Stan data object, defaulting to an empty object.
        seed = Seed for model construction, including transformed data.
        dependencyDirs = Additional Windows DLL directories for model loading.
    Throws: Exception on load, version, symbol or model initialization failure.
    */
    this(string library, string json = "{}", uint seed = 1, string[] dependencyDirs = null)
    {
        json = json.strip;
        if (!json.length || json[0] != '{' || json[$-1] != '}' || json.hasNul)
            throw new Exception("Expected an in-memory JSON object");
        synchronized { lib.open(library, dependencyDirs); }
        scope(failure) close();
        TinyStanError* error;
        handle = lib.create(json.toStringz, seed, null, &error);
        if (!handle || error) fail(error, "TinyStan model construction failed");
    }

    /// Releases the model before its library. Repeated calls are safe.
    void close() nothrow
    {
        if (handle)
        {
            // The pinned implementation only deletes the C++ model.
            alias Destroy = extern(C) void function(TinyStanModel*) nothrow;
            (cast(Destroy)lib.destroy)(handle);
            handle = null;
        }
        lib.close();
    }
    ~this() { close(); }

    /// Returns: Embedded Stan major, minor and patch version.
    @property int[3] stanVersion() const @safe pure nothrow @nogc { return lib.stanVersion; }
    private void fail(TinyStanError* error, string fallback)
    {
        if (!error) throw new StanException(fallback, 0);
        scope(exit) lib.freeError(error);
        auto message = lib.message(error);
        throw new StanException(message ? message.fromStringz.idup : fallback, lib.errorType(error));
    }
}

unittest
{
    Model empty;
    static assert(!__traits(compiles, { auto duplicate = empty; }));
    empty.close();
}
