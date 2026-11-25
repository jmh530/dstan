/// Resource ownership and synchronous inference.
module dstan.model;

import dstan.internal.library;
import dstan.result;
import dstan.internal.tinystan;
import std.string : fromStringz, split, strip, toStringz;

/// Diagonal-metric NUTS settings. Warmup is discarded.
struct SampleOptions
{
    uint seed = 1;       /// Sampling RNG seed. Chain IDs start at one.
    int warmup = 1000;   /// Adaptation iterations per chain.
    int draws = 1000;    /// Retained iterations per chain.
    int chains = 1;      /// Number of chains in a single TinyStan call. Must be positive.
}

/// TinyStan status translated into D-owned exception data.
class StanException : Exception
{
    int category; /// TinyStan error type (generic=0, configuration=1, interrupt=2).
    package this(string msg, int category) { super(msg); this.category = category; }
}

private immutable string[] diagnostics = ["lp__", "accept_stat__", "stepsize__", "treedepth__",
    "n_leapfrog__", "divergent__", "energy__"];

/// A scoped, noncopyable owner of one initialized Stan model and its library.
struct Model
{
    @disable this(this);
    private Library lib;
    private TinyStanModel* handle;
    private immutable(string)[] columns;

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
        auto names = lib.names(handle).fromStringz.idup;
        columns = diagnostics;
        if (names.length) columns ~= names.split(",").idup;
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
    /// Returns: D-owned diagnostic and flattened model column names in draw order.
    @property immutable(string)[] names() const @safe pure nothrow @nogc { return columns; }

    /**
    Params: options = NUTS sampling settings.
    Returns: Required number of doubles for the requested saved draws.
    Throws: Exception for a closed model, invalid settings or overflow.
    */
    size_t requiredCount(SampleOptions options = SampleOptions.init) const @safe
    {
        if (!handle) throw new Exception("Closed Stan model");
        if (options.warmup < 0 || options.draws <= 0 || options.warmup > int.max - options.draws)
            throw new Exception("Invalid iteration count");
        return checkedCount(options.chains, options.draws, columns.length);
    }

    /**
    Params: options = NUTS sampling settings.
    Returns: D-owned draws with shape [options.chains, options.draws, names.length].
    Throws: Exception on invalid settings or inference failure.
    */
    Draws sample(SampleOptions options = SampleOptions.init) @safe
    {
        auto values = new double[requiredCount(options)];
        sampleInto(values, options);
        return Draws(values, columns, [cast(size_t)options.chains, cast(size_t)options.draws, columns.length]);
    }

    /**
    Params:
        output = Contiguous writable doubles, exclusively available during this call.
            Writes the required prefix in chain/draw/column order. Failure may leave partial draws.
        options = NUTS sampling settings.
    Returns: Number of doubles written.
    Throws: Exception on invalid capacity, settings or inference failure.
    */
    size_t sampleInto(scope double[] output, SampleOptions options = SampleOptions.init) @safe
    {
        auto count = requiredCount(options);
        if (output.length < count) throw new Exception("Insufficient output buffer capacity");
        if ((cast(size_t) output.ptr) % double.alignof)
            throw new Exception("Misaligned output buffer");
        runSampler(output[0 .. count], options);
        return count;
    }

    // Only called with the verified capacity, a live model and exclusive access
    // to output. The pinned library joins its workers and releases its writers
    // before returning. Loading a trusted model library is an @system operation.
    private void runSampler(scope double[] output, SampleOptions options) @trusted
    {
        import core.memory : GC;
        // Keep the slice pointer visible even while workers hold it in C++ memory.
        auto storage = output.ptr;
        GC.addRange(&storage, storage.sizeof);
        scope(exit) GC.removeRange(&storage);
        TinyStanError* error;
        int status;
        // Upstream installs process-wide interrupt handlers. All DStan calls share this lock.
        synchronized
        {
            version (Windows)
                if (options.chains > 1)
                    lib.retainSamplingRuntime();
            status = lib.sample(handle, options.chains, null, options.seed, 1, 2.0,
                options.warmup, options.draws, diagonal, null, true, 0.8, 0.05,
                0.75, 10, 75, 50, 25, false, 1.0, 0, 10, 0, 1,
                storage, output.length, null, null, &error);
        }
        if (status || error) fail(error, "TinyStan sampling failed");
    }

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
    import std.exception : assertThrown;
    Model empty;
    static assert(!__traits(compiles, { auto duplicate = empty; }));
    empty.close();
    assertThrown(empty.requiredCount());
}
