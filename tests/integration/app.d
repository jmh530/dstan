import dstan;
import std.exception : enforce, assertThrown;
import std.string : indexOf;
import std.math : abs, sqrt;
import std.path : absolutePath, buildPath;
import std.stdio : writefln;
import std.process : environment, execute;

private void checkIncludedSource()
{
    import std.file : mkdir, rmdirRecurse, tempDir, write;
    import std.string : replace;
    import std.uuid : randomUUID;

    auto dir = buildPath(tempDir, "dstan-includes-" ~ randomUUID().toString);
    version (Windows) dir = dir.replace("\\", "/");
    mkdir(dir);
    scope(exit) rmdirRecurse(dir);
    auto source = dir ~ "/included.stan";
    auto prior = dir ~ "/prior.stan";
    write(source, "parameters { real x; }\nmodel {\n#include outer.stan\n}\n");
    write(dir ~ "/outer.stan", "#include prior.stan\n");
    write(prior, "x ~ std_normal();\n");
    // Load a Make setting after the upstream rules without editing the installation.
    auto local = dir ~ "/flags.mk";
    write(local, "STANCFLAGS = --include-paths=" ~ dir ~ "\n");
    CompileOptions options;
    options.makeArgs = ["-f", "Makefile", "-f", local];
    compileModel(source, options);
    write(prior, "x ~ ; // changed nested prior\n");
    try
    {
        compileModel(source, options);
        assert(0, "Reused a model built with a different nested include");
    }
    catch (Exception e) { assert(e.msg.indexOf("changed nested prior") >= 0, e.msg); }
}

private void checkChains(ref Model model, int threads)
{
    import std.file : write;

    SampleOptions options;
    options.seed = 321;
    options.draws = 4000;
    options.chains = 2;
    options.threads = threads;
    auto fit = model.sample(options);
    assert(fit.shape == [2, 4000, 8]);
    assert(fit.names == model.names);
    assert(fit.savedWarmup == 0);
    const stride = options.draws * fit.names.length;
    assert(model.requiredCount(options) == 2 * stride);
    auto storage = new double[2 * stride + 2];
    storage[0] = storage[$-1] = -123;
    auto used = model.sampleInto(storage[1 .. $-1], options);
    assert(used == 2 * stride && storage[1 .. $-1] == fit.values);
    assert(storage[0] == -123 && storage[$-1] == -123);
    assertThrown(model.sampleInto(storage[1 .. $-2], options));
    assert(fit.values[0 .. stride] != fit.values[stride .. $]);
    foreach (chain; 0 .. options.chains)
    {
        double mean = 0;
        foreach (draw; 0 .. options.draws)
        {
            auto theta = fit[chain, draw, 7];
            assert(theta == storage[1 + chain * stride + draw * 8 + 7]);
            mean += theta;
        }
        assert(abs(mean / options.draws - 2.5) < 0.1);
    }
    auto prefix = environment.get("DSTAN_REFERENCE_DRAWS", "");
    auto suffix = threads == 1 ? ".serial" : ".parallel";
    if (prefix.length) write(prefix ~ suffix, fit.values);

    options.saveWarmup = true;
    auto full = model.sample(options);
    assert(full.shape == [2, 5000, 8] && full.savedWarmup == options.warmup);
    auto fullStride = full.shape[1] * full.names.length;
    foreach (chain; 0 .. full.shape[0])
    {
        auto start = chain * fullStride + full.savedWarmup * full.names.length;
        assert(full.values[start .. start + stride] == fit.values[chain * stride .. (chain + 1) * stride]);
    }
    assert(model.requiredCount(options) == full.values.length);
    auto guarded = new double[full.values.length + 2];
    guarded[0] = guarded[$-1] = -123;
    assertThrown(model.sampleInto(guarded[1 .. $-2], options));
    assert(model.sampleInto(guarded[1 .. $-1], options) == full.values.length);
    assert(guarded[1 .. $-1] == full.values && guarded[0] == -123 && guarded[$-1] == -123);
    if (prefix.length) write(prefix ~ suffix ~ ".warmup", full.values);

    options.warmup = 100;
    options.draws = 20;
    options.stepSize = 0.25;
    auto smallerStep = model.sample(options);
    assert(smallerStep.shape == [2, 120, 8]);
    if (prefix.length) write(prefix ~ suffix ~ ".step-size", smallerStep.values);
    auto adaptation = options;
    adaptation.adaptDelta = 0.9;
    auto adapted = model.sample(adaptation);
    if (prefix.length) write(prefix ~ suffix ~ ".adapt-delta", adapted.values);
    foreach (invalid; [-0.1, 0.0, 1.0, 1.1, double.nan, double.infinity, -double.infinity])
    {
        adaptation.adaptDelta = invalid;
        assertThrown(model.requiredCount(adaptation));
    }
    adaptation.adapt = false;
    adaptation.adaptDelta = double.nan; // Ignored when adaptation is disabled.
    auto fixed = model.sample(adaptation);
    assert(fixed.shape == smallerStep.shape && fixed.savedWarmup == options.warmup);
    assert(fixed.names[2] == "stepsize__");
    foreach (chain; 0 .. fixed.shape[0])
        foreach (row; 0 .. fixed.shape[1])
            assert(fixed[chain, row, 2] == options.stepSize);
    if (prefix.length) write(prefix ~ suffix ~ ".no-adapt", fixed.values);
    auto depthOptions = options;
    depthOptions.maxTreeDepth = 1;
    auto capped = model.sample(depthOptions);
    assert(capped.shape == smallerStep.shape && capped.names[3] == "treedepth__");
    foreach (chain; 0 .. capped.shape[0])
    {
        bool defaultExceededCap;
        foreach (row; 0 .. capped.shape[1])
        {
            assert(capped[chain, row, 3] <= depthOptions.maxTreeDepth);
            if (smallerStep[chain, row, 3] > depthOptions.maxTreeDepth)
                defaultExceededCap = true;
        }
        assert(defaultExceededCap);
    }
    if (prefix.length) write(prefix ~ suffix ~ ".depth", capped.values);
    foreach (invalid; [0, -1, 32, int.max])
    {
        depthOptions.maxTreeDepth = invalid;
        assertThrown(model.requiredCount(depthOptions));
    }
    options.stepSizeJitter = 0.25;
    auto jittered = model.sample(options);
    assert(jittered.shape == smallerStep.shape && jittered.values != smallerStep.values);
    if (prefix.length) write(prefix ~ suffix ~ ".jitter", jittered.values);
    foreach (invalid; [-0.01, 1.0, 1.01, double.nan, double.infinity, -double.infinity])
    {
        options.stepSizeJitter = invalid;
        assertThrown(model.requiredCount(options));
    }
    options.stepSizeJitter = 0;
    foreach (invalid; [0.0, -1.0, double.nan, double.infinity, -double.infinity])
    {
        options.stepSize = invalid;
        assertThrown(model.requiredCount(options));
    }
    options.stepSize = 1.0;
    options.initialValues = [`{"theta":-2}`, `{"theta":2}`];
    options.initRadius = 0.75;
    auto initialized = model.sample(options);
    assert(initialized.shape == [2, 120, 8]);
    if (prefix.length) write(prefix ~ suffix ~ ".initialized", initialized.values);
    options.initialValues = [`{"theta":-2}`];
    auto sharedInit = model.sample(options);
    options.initialValues = [`{"theta":-2}`, `{"theta":-2}`];
    assert(sharedInit.values == model.sample(options).values);
    // theta is unconstrained, so radius zero should reproduce an explicit zero start.
    options.initRadius = 0;
    options.initialValues = ["{}"];
    auto zero = model.sample(options);
    options.initialValues = [`{"theta":0}`];
    assert(zero.values == model.sample(options).values);
    if (prefix.length) write(prefix ~ suffix ~ ".zero-radius", zero.values);
    options.initialValues = [`{broken}`];
    assertThrown!StanException(model.sample(options));
    options.initialValues = ["{}", "{}", "{}"];
    assertThrown(model.requiredCount(options));
    options.initialValues = null;
    foreach (invalid; [-1.0, double.nan, double.infinity])
    {
        options.initRadius = invalid;
        assertThrown(model.requiredCount(options));
    }
}

private void checkMetrics(ref Model model)
{
    import std.conv : to;
    import std.file : write;

    SampleOptions options;
    options.seed = 321;
    options.chains = 2;
    options.warmup = 200;
    options.draws = 20;
    options.saveWarmup = true;
    auto prefix = environment.get("DSTAN_REFERENCE_DRAWS", "");
    assert(model.numFreeParams == 2);
    foreach (metric; [Metric.unit, Metric.diagonal, Metric.dense])
        foreach (adapt; [false, true])
        {
            options.metric = metric;
            options.adapt = adapt;
            options.initialInverseMetric = null;
            auto fit = model.sample(options);
            assert(fit.shape == [2, 220, 10] && fit.savedWarmup == options.warmup);
            foreach (chain; 0 .. fit.shape[0])
                foreach (row; 0 .. fit.shape[1])
                    assert(abs(fit[chain, row, 7] + fit[chain, row, 8] - fit[chain, row, 9]) < 1e-10);
            if (prefix.length)
                write(prefix ~ "." ~ metric.to!string ~ (adapt ? ".adapted" : ".fixed"), fit.values);
            if (metric == Metric.unit) continue;
            options.initialInverseMetric = metric == Metric.diagonal
                ? [0.5, 2.0, 1.5, 0.75]
                : [0.5, 0.25, 0.25, 2.0, 1.5, -0.125, -0.125, 0.75];
            auto initialized = model.sample(options);
            assert(initialized.shape == fit.shape && initialized.values != fit.values);
            if (prefix.length)
                write(prefix ~ "." ~ metric.to!string ~ (adapt ? ".adapted" : ".fixed") ~ ".initial", initialized.values);
        }

    options.metric = Metric.unit;
    assertThrown(model.requiredCount(options));
    options.metric = Metric.diagonal;
    foreach (wrongLength; [2, 3, 5])
    {
        options.initialInverseMetric = new double[wrongLength];
        assertThrown(model.requiredCount(options));
    }
    options.initialInverseMetric = [1.0, 1.0, 1.0, -0.5];
    assertThrown!StanException(model.sample(options));
    options.metric = Metric.dense;
    options.initialInverseMetric = [1.0, 0, 0, 1, 1, 2, 2, 1];
    assertThrown!StanException(model.sample(options));
    options.initialInverseMetric = [1.0, 0, 0, 1, 1, 0.5, 0, 1];
    assertThrown!StanException(model.sample(options));
    options.initialInverseMetric = [1.0, 0, 0, 1, double.infinity, 0, 0, 1];
    assertThrown(model.sample(options));
    auto storage = new double[model.requiredCount(options)];
    storage[$-8 .. $] = [1.0, 0, 0, 1, 1, 0, 0, 1];
    options.initialInverseMetric = storage[$-8 .. $];
    assertThrown(model.sampleInto(storage, options));
}

private void checkAdaptationResults(ref Model model)
{
    import std.conv : to;
    import std.file : write;
    SampleOptions options;
    options.seed = 321;
    options.chains = 2;
    options.warmup = 200;
    options.draws = 20;
    auto prefix = environment.get("DSTAN_REFERENCE_DRAWS", "");
    foreach (metric; [Metric.unit, Metric.diagonal, Metric.dense])
    {
        options.metric = metric;
        auto fit = model.sampleWithAdaptation(options);
        assert(fit.draws.savedWarmup == 0 && fit.draws.shape == [2, 20, 10]);
        assert(fit.draws.values == model.sample(options).values);
        assert(fit.stepSizes.length == options.chains);
        foreach (chain, step; fit.stepSizes)
            foreach (row; 0 .. options.draws)
                assert(step == fit.draws[chain, row, 2]); // stepsize__, zero jitter
        const expected = metric == Metric.unit ? 0 : metric == Metric.diagonal ? 4 : 8;
        assert(fit.inverseMetric.length == expected);
        auto reuse = options;
        reuse.adapt = false;
        reuse.initialInverseMetric = fit.inverseMetric;
        auto fixed = model.sample(reuse);
        assert(fixed.shape == fit.draws.shape);
        if (prefix.length)
        {
            auto base = prefix ~ "." ~ metric.to!string;
            write(base ~ ".step-sizes", fit.stepSizes);
            if (expected) write(base ~ ".inverse-metric", fit.inverseMetric);
            write(base ~ ".reused", fixed.values);
        }
    }
    options.adapt = false;
    assertThrown(model.sampleWithAdaptation(options));
}

void main(string[] args)
{
    if (args.length == 3 && args[1] == "--compile-interrupt")
    {
        CompileOptions options;
        options.stancArgs = ["--allow-undefined"];
        compileModel(args[2], options);
        return;
    }
    if (args.length >= 4 && args[1] == "--interrupt")
    {
        import interrupts : checkInterrupt;
        import std.conv : to;
        checkInterrupt(args[2], args[3].to!int, args[4 .. $]);
        return;
    }
    if (args.length >= 3 && args[1] == "--parallel")
    {
        auto model = Model(args[2], `{"mu":2.5}`, 123, args[3 .. $]);
        checkChains(model, 2);
        return;
    }
    enforce(args.length >= 2, "Pass tests/models and optionally Windows DLL directories");
    auto root = args[1].absolutePath;
    checkIncludedSource();
    CompileOptions missing;
    missing.root = buildPath(root, "missing-installation");
    assertThrown(compileModel(buildPath(root, "normal.stan"), missing));
    try
    {
        compileModel(buildPath(root, "invalid.stan"));
        assert(0, "Expected stanc to reject invalid source");
    }
    catch (Exception ex)
    {
        assert(ex.msg.indexOf("Syntax error") >= 0, ex.msg);
    }
    auto normal = compileModel(buildPath(root, "normal.stan"));
    auto structured = compileModel(buildPath(root, "structured.stan"));
    auto dirs = args[2 .. $];
    version (Windows)
    {
        import std.file : copy, mkdir, rmdirRecurse, tempDir;
        import std.uuid : randomUUID;
        auto dir = buildPath(tempDir, "dstan loader " ~ randomUUID().toString);
        mkdir(dir);
        scope(exit) rmdirRecurse(dir);
        auto spacedPath = buildPath(dir, "normal model.so");
        copy(normal, spacedPath);
        // The dedicated Windows run keeps TBB outside the default DLL directories.
        if (dirs.length) assertThrown(Model(spacedPath, `{"mu":0}`));
        auto spaced = Model(spacedPath, `{"mu":0}`, 1, dirs);
        assert(spaced.names[$-1] == "theta");
        spaced.close();
    }
    assertThrown!StanException(Model(normal, `{"mu":[]}`, 1, dirs));
    assertThrown(Model(normal, "bad.json", 1, dirs));
    assertThrown(Model(normal ~ ".missing", "{}", 1, dirs));
    auto model = Model(normal, `{"mu":2.5}`, 123, dirs);
    auto second = Model(structured, "{}", 123, dirs);
    assert(model.stanVersion == [2, 39, 0]);
    assert(model.names == ["lp__", "accept_stat__", "stepsize__", "treedepth__",
        "n_leapfrog__", "divergent__", "energy__", "theta"]);
    assert(second.names[7 .. $] == ["x.1", "x.2", "total"]);
    SampleOptions options;
    options.warmup = -1;
    assertThrown(model.requiredCount(options));
    options.draws = 1;
    options.warmup = 0;
    options.saveWarmup = true;
    assert(model.requiredCount(options) == model.names.length);
    options.warmup = int.max / cast(int) model.names.length;
    assertThrown(model.requiredCount(options));
    options.warmup = int.max;
    assertThrown(model.requiredCount(options));
    options.saveWarmup = false;
    options.warmup = 1000;
    options.threads = -2;
    assertThrown(model.requiredCount(options));
    options.threads = 0;
    assertThrown(model.requiredCount(options));
    options.threads = 1;
    options.draws = 4000;
    options.seed = 321;
    auto result = model.sample(options);
    assert(result.shape == [1, 4000, 8]);
    auto storage = new double[model.requiredCount(options) + 2];
    storage[0] = storage[$-1] = -123;
    auto buf = storage[1 .. $-1];
    assert(model.sampleInto(buf, options) == buf.length);
    assert(buf.ptr == &storage[1] && buf == result.values);
    assert(storage[0] == -123 && storage[$-1] == -123);
    assertThrown(model.sampleInto(buf[1 .. $], options));
    checkChains(model, 1);
    // TBB keeps the first pool size. Start a new process to exercise two threads.
    import std.file : thisExePath;
    auto parallel = execute([thisExePath(), "--parallel", normal] ~ dirs);
    enforce(parallel.status == 0, parallel.output);
    model.close();
    model.close();
    assertThrown(model.sample(options));
    double mean = 0, variance = 0;
    foreach (i; 0 .. options.draws) mean += result[0, i, 7];
    mean /= options.draws;
    foreach (i; 0 .. options.draws) variance += (result[0, i, 7] - mean) ^^ 2;
    auto sd = sqrt(variance / (options.draws - 1));
    assert(abs(mean - 2.5) < 0.1 && abs(sd - 1) < 0.1);
    checkMetrics(second);
    checkAdaptationResults(second);
    auto reopened = Model(normal, `{"mu":1}`, 1, dirs);
    assert(reopened.names == result.names);
    writefln("shape=%s mean=%.12f sd=%.12f", result.shape, mean, sd);
}
