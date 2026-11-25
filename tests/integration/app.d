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
    if (prefix.length)
        write(prefix ~ (threads == 1 ? ".serial" : ".parallel"), fit.values);
}

void main(string[] args)
{
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
    auto multi = second.sample();
    foreach (i; 0 .. multi.shape[1])
        assert(abs(multi[0, i, 7] + multi[0, i, 8] - multi[0, i, 9]) < 1e-10);
    auto reopened = Model(normal, `{"mu":1}`, 1, dirs);
    assert(reopened.names == result.names);
    writefln("shape=%s mean=%.12f sd=%.12f", result.shape, mean, sd);
}
