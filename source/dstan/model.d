/// Resource ownership and synchronous inference.
module dstan.model;

import dstan.internal.library;
import dstan.result;
import dstan.internal.tinystan;
import std.string : fromStringz, split, strip, toStringz;

/// Euclidean metric used by NUTS on the unconstrained parameter scale.
enum Metric
{
    diagonal = dstan.internal.tinystan.diagonal, /// Adapt a separate scale for each parameter when adaptation is enabled.
    unit = dstan.internal.tinystan.unit,         /// Keep the identity metric even during adaptation.
    dense = dstan.internal.tinystan.dense,       /// Adapt a full matrix when adaptation is enabled.
}

/// NUTS settings. Warmup is discarded by default.
struct SampleOptions
{
    uint seed = 1;       /// Sampling RNG seed. Chain IDs start at one.
    int warmup = 1000;   /// Warmup iterations per chain, including when adaptation is disabled.
    int draws = 1000;    /// Post-warmup draws per chain.
    int chains = 1;      /// Number of chains in a single TinyStan call. Must be positive.
    int threads = 1;     /// Requested Stan thread-pool size. Set on first use; -1 requests available cores.
    bool saveWarmup = false;  /// Keep warmup rows at the beginning of each chain.

    const(string)[] initialValues; /// In-memory JSON objects on the constrained scale. Supply zero, one shared object, or one per chain.
    double initRadius = 2.0;       /// Finite nonnegative radius for unspecified coordinates on the unconstrained scale.
    double stepSize = 1.0;         /// Initial NUTS step size. Must be finite and positive.

    /// Maximum fractional variation in step size per trajectory. Zero disables jitter.
    /// Use a finite value in [0, 1). See $(LINK2 https://github.com/stan-dev/stan/issues/3462, Stan issue 3462).
    double stepSizeJitter = 0.0;

    /// Maximum tree depth per NUTS transition, from 1 through 31 inclusive. Trees may stop growing earlier.
    int maxTreeDepth = 10;

    /// Adapt the step size and any diagonal or dense metric during warmup.
    bool adapt = true;

    /// Target average acceptance probability during adaptation. Must be in (0, 1) when adapt is true.
    double adaptDelta = 0.8;

    /// NUTS metric for all chains. Dense supports up to 46,340 unconstrained parameters in TinyStan 0.3.3.
    Metric metric = Metric.diagonal;

    /// Initial inverse mass matrix on the unconstrained scale. Empty selects identity.
    /// Supply one contiguous block per chain: p positive values for a diagonal metric,
    /// or a symmetric positive-definite p-by-p matrix in column-major order for a dense
    /// metric, where p is Model.numFreeParams. Entries must be finite. Keep the slice
    /// readable and unchanged until sampling returns. Unit metrics require empty input.
    const(double)[] initialInverseMetric;

    private int savedWarmup() const @safe pure nothrow @nogc
    {
        return saveWarmup ? warmup : 0;
    }

    private int savedRows() const @safe
    {
        if (warmup < 0 || draws <= 0 || warmup > int.max - draws)
            throw new Exception("Invalid iteration count");
        return savedWarmup() + draws;
    }
}

/// Settings for one L-BFGS run without the parameter-transform Jacobian adjustment.
struct OptimizeOptions
{
    uint seed = 1;            /// RNG seed for initialization and generated quantities.
    string initialValues;     /// In-memory JSON object of constrained initial values. Empty uses Stan's random start.
    int maxIterations = 2000; /// Positive iteration limit. Stan may stop earlier on convergence.
}

private string checkedJsonObject(string json) @safe
{
    json = json.strip;
    if (!json.length || json[0] != '{' || json[$-1] != '}' || json.hasNul)
        throw new Exception("Expected an in-memory JSON object");
    return json;
}

// TinyStan parses each object itself. Its delimiter must never occur as a
// literal byte inside one object, even for a shared initialization.
private string encodeInitialValues(const(string)[] values, char separator) @safe
{
    import std.array : join;
    import std.string : indexOf;

    foreach (value; values)
    {
        checkedJsonObject(value);
        if (value.indexOf(separator) >= 0)
            throw new Exception("TinyStan separator in an initial-value object");
    }
    if (values.length == 1) return values[0];
    return values.join([separator]);
}

unittest
{
    import std.exception : assertThrown;
    enum sep = '\x1c';
    assert(encodeInitialValues(["{}", " {} "], sep) == "{}\x1c {} ");
    foreach (invalid; ["", "init.json", "[]", "{\0}", "{}\x1c{}"])
        assertThrown(encodeInitialValues([invalid], sep));
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
    private size_t freeParamCount;
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
        json = checkedJsonObject(json);
        synchronized { lib.open(library, dependencyDirs); }
        scope(failure) close();
        TinyStanError* error;
        handle = lib.create(json.toStringz, seed, null, &error);
        if (!handle || error) fail(error, "TinyStan model construction failed");
        freeParamCount = lib.numFreeParams(handle);
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
    /// Returns: Number of unconstrained real parameters per chain, used to size inverse metrics.
    @property size_t numFreeParams() const @safe pure nothrow @nogc { return freeParamCount; }

    /**
    Params: options = L-BFGS settings, defaulting to OptimizeOptions.init.
    Returns: One D-owned row containing the log objective, Stan's numeric termination
        code and the flattened constrained model outputs, including generated quantities.
        Code 40 indicates that the iteration limit was reached.
    Throws: Exception for a closed model or invalid settings. StanException on upstream failure.
    */
    OptimizationResult optimize(OptimizeOptions options = OptimizeOptions.init) @safe
    {
        if (!handle) throw new Exception("Closed Stan model");
        if (options.maxIterations <= 0) throw new Exception("Optimization iteration limit must be positive");
        if (options.initialValues.length)
            options.initialValues = checkedJsonObject(options.initialValues);

        static immutable string[] prefix = ["lp__", "converged__"];
        auto names = prefix ~ columns[diagnostics.length .. $];
        auto values = new double[names.length];
        runOptimizer(values, options);
        return OptimizationResult(values, names);
    }

    // The pinned L-BFGS service writes one vector of this exact width. Both
    // GC pointers stay rooted until its synchronous writer and JSON reader return.
    private void runOptimizer(scope double[] output, OptimizeOptions options) @trusted
    {
        import core.memory : GC;
        struct Buffers { double* output; const(char)* inits; }
        Buffers buffers;
        buffers.output = output.ptr;
        if (options.initialValues.length) buffers.inits = options.initialValues.toStringz;
        GC.addRange(&buffers, buffers.sizeof);
        scope(exit) GC.removeRange(&buffers);
        TinyStanError* error;
        int status;
        synchronized (typeid(Model))
        {
            status = lib.optimize(handle, buffers.inits, options.seed, 1, 2.0, lbfgs,
                options.maxIterations, false, 5, 0.001, 1e-12, 1e4, 1e-8, 1e7, 1e-8,
                0, 1, buffers.output, output.length, &error);
        }
        if (status || error) fail(error, "TinyStan optimization failed");
    }

    // Called after checking the dense dimension. TinyStan uses a signed int for
    // each per-chain block and accepts pointers without buffer lengths.
    private size_t metricCount(SampleOptions options) const @safe
    {
        if (options.metric == Metric.unit) return 0;
        const block = options.metric == Metric.dense ? numFreeParams * numFreeParams : numFreeParams;
        if (options.chains <= 0 || block > int.max || block > size_t.max / double.sizeof / options.chains)
            throw new Exception("Inverse metric dimensions exceed supported limits");
        return options.chains * block;
    }

    /**
    Params: options = NUTS sampling settings.
    Returns: Required number of doubles for the requested saved draws.
    Throws: Exception for a closed model, invalid settings or overflow.
    */
    size_t requiredCount(SampleOptions options = SampleOptions.init) const @safe
    {
        if (!handle) throw new Exception("Closed Stan model");
        import std.math : isFinite;
        if (options.metric != Metric.unit && options.metric != Metric.diagonal && options.metric != Metric.dense)
            throw new Exception("Invalid NUTS metric");
        // TinyStan 0.3.3 squares this dimension in a signed int even with null metric buffers.
        if (options.metric == Metric.dense && numFreeParams > 46_340)
            throw new Exception("Dense NUTS metric exceeds 46,340 unconstrained parameters");
        if (options.initialInverseMetric.length)
        {
            if (options.metric == Metric.unit)
                throw new Exception("Unit metrics require an empty initial inverse metric");
            if (options.initialInverseMetric.length != metricCount(options))
                throw new Exception("Supply exactly one initial inverse metric per chain");
            if ((cast(size_t) options.initialInverseMetric.ptr) % double.alignof)
                throw new Exception("Misaligned initial inverse metric");
        }
        // A full depth-d transition takes 2^d - 1 leapfrog steps. Stan 2.39.0
        // accumulates that count in a signed 32-bit int.
        if (options.maxTreeDepth < 1 || options.maxTreeDepth > 31)
            throw new Exception("Maximum NUTS tree depth must be between 1 and 31 inclusive");
        if (!isFinite(options.stepSize) || options.stepSize <= 0)
            throw new Exception("Initial step size must be finite and positive");
        // Stan 2.39.0's step-size adaptation setter silently ignores both endpoints.
        if (options.adapt && !(options.adaptDelta > 0 && options.adaptDelta < 1))
            throw new Exception("Target acceptance probability must be in (0, 1)");
        // Stan 2.39.0 silently ignores 1.0. Revisit the exclusive upper bound
        // when updating Stan: https://github.com/stan-dev/stan/issues/3462
        if (!(options.stepSizeJitter >= 0 && options.stepSizeJitter < 1))
            throw new Exception("Step-size jitter must be in [0, 1). Stan 2.39.0 silently ignores 1.0.");
        if (!isFinite(options.initRadius) || options.initRadius < 0)
            throw new Exception("Initialization radius must be finite and nonnegative");
        if (options.initialValues.length > 1 && options.initialValues.length != options.chains)
            throw new Exception("Supply zero initial-value objects, one shared object, or one per chain");
        if (options.threads == 0 || options.threads < -1) throw new Exception("Invalid Stan thread count");
        return checkedCount(options.chains, options.savedRows(), columns.length);
    }

    /**
    Params: options = NUTS sampling settings.
    Returns: D-owned draws with shape [options.chains, saved rows per chain, names.length].
        Each chain contains options.draws posterior rows preceded by any saved warmup rows.
    Throws: Exception on invalid settings or inference failure.
    */
    Draws sample(SampleOptions options = SampleOptions.init) @safe
    {
        return sampleAllocated(options, requiredCount(options), null, null);
    }

    /**
    Params: options = NUTS settings, defaulting to SampleOptions.init. Adaptation must be enabled.
    Returns: D-owned draws plus one final nominal step size per chain and a flat inverse metric
        in chain order. Each diagonal block has numFreeParams entries and each dense block has
        numFreeParams squared entries in column-major order. Unit metrics return an empty slice.
        Tuning values are available whether or not warmup draws are saved.
    Throws: Exception for a closed model, invalid settings, size overflow, inference failure,
        or a chain that did not return a valid final step size.
    */
    AdaptationResult sampleWithAdaptation(SampleOptions options = SampleOptions.init) @safe
    {
        import std.math : isFinite;
        if (!options.adapt) throw new Exception("Adaptation results require adaptation to be enabled");
        const count = requiredCount(options);
        const metricSize = metricCount(options);
        // The checked draw buffer already holds at least one double per chain.
        auto stepSizes = new double[options.chains];
        double[] inverseMetric;
        if (metricSize) inverseMetric = new double[metricSize];
        auto draws = sampleAllocated(options, count, stepSizes, inverseMetric);
        // Stan can return early during step-size initialization without writing the
        // structured state. An unwritten D double retains its NaN initializer.
        foreach (step; stepSizes)
            if (!isFinite(step) || step <= 0)
                throw new Exception("Stan did not return a valid final step size for every chain");
        return AdaptationResult(draws, stepSizes, inverseMetric);
    }

    private Draws sampleAllocated(SampleOptions options, size_t count, scope double[] stepSizes,
                                   scope double[] inverseMetric) @safe
    {
        auto values = new double[count];
        runSampler(values, options, stepSizes, inverseMetric);
        return Draws(values, columns,
            [cast(size_t)options.chains, cast(size_t)options.savedRows(), columns.length],
            cast(size_t)options.savedWarmup());
    }

    /**
    Params:
        output = Contiguous writable doubles, exclusively available during this call.
            Writes the required prefix in chain/draw/column order. The prefix must be disjoint from
            initialInverseMetric. Saved warmup rows come first within each chain.
            Failure may leave partial draws.
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
        auto draws = output[0 .. count];
        if (options.initialInverseMetric.length)
        {
            import std.array : overlap;
            if (overlap(draws, options.initialInverseMetric).length)
                throw new Exception("Initial inverse metric overlaps the draw buffer");
        }
        runSampler(draws, options, null, null);
        return count;
    }

    // Draw capacity is checked. Any adaptation outputs are disjoint fresh allocations
    // sized for all chains and never exposed until success. The pinned library joins its workers and releases its writers
    // before returning. Loading a trusted model library is an @system operation.
    private void runSampler(scope double[] output, SampleOptions options, scope double[] stepSizes,
                            scope double[] inverseMetric) @trusted
    {
        import core.memory : GC;
        import std.math : isFinite;
        foreach (value; options.initialInverseMetric)
            if (!isFinite(value)) throw new Exception("Initial inverse metric entries must be finite");
        // Root borrowed buffers for the synchronous call. TinyStan's metric readers
        // hold pointers into the checked slice until they copy the values into Stan.
        struct Buffers
        {
            double* output;
            const(char)* inits;
            const(double)* metric;
            double* stepSizes;
            double* inverseMetric;
        }
        Buffers buffers;
        buffers.output = output.ptr;
        buffers.stepSizes = stepSizes.ptr;
        buffers.inverseMetric = inverseMetric.ptr;
        if (options.initialInverseMetric.length)
            buffers.metric = options.initialInverseMetric.ptr;
        if (options.initialValues.length)
            buffers.inits = encodeInitialValues(options.initialValues, lib.separator).toStringz;
        GC.addRange(&buffers, buffers.sizeof);
        scope(exit) GC.removeRange(&buffers);
        TinyStanError* error;
        int status;
        // Upstream installs process-wide interrupt handlers. All DStan calls share this lock.
        synchronized (typeid(Model))
        {
            version (Windows)
                if (options.chains > 1 || options.threads != 1)
                    lib.retainSamplingRuntime();
            status = lib.sample(handle, options.chains, buffers.inits, options.seed, 1, options.initRadius,
                options.warmup, options.draws, cast(TinyStanMetric) options.metric, buffers.metric,
                options.adapt, options.adaptDelta, 0.05,
                0.75, 10, 75, 50, 25, options.saveWarmup, options.stepSize, options.stepSizeJitter,
                options.maxTreeDepth, 0, options.threads,
                buffers.output, output.length, buffers.stepSizes, buffers.inverseMetric, &error);
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
