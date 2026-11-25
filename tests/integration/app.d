import dstan;
import std.exception : enforce, assertThrown;
import std.string : indexOf;
import std.math : abs, sqrt;
import std.path : absolutePath, buildPath;
import std.stdio : writefln;

void main(string[] args)
{
    enforce(args.length >= 2, "Pass tests/models and optionally Windows DLL directories");
    auto root = args[1].absolutePath;
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
