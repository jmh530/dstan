/// Compile Stan source with the installed TinyStan Makefile.
module dstan.compile;

import std.file;
import std.path;
import std.process : Config, environment, execute;
import std.string : join, replace, splitLines;
import std.algorithm : canFind;

/// Options forwarded to the pinned upstream build.
struct CompileOptions
{
    string root;            /// TinyStan root, defaulting to TINYSTAN or the user's versioned cache.
    string make;            /// Make executable, defaulting to MAKE or make.
    string[] makeArgs;      /// Individual arguments to GNU Make.
    string[] stancArgs;     /// Explicit stanc flags replace Make's STANCFLAGS; empty inherits upstream settings.
}

private string installationRoot(string configured)
{
    if (configured.length) return configured;
    auto installed = environment.get("TINYSTAN", "");
    if (installed.length) return installed;
    version (Windows)
    {
        auto home = environment.get("USERPROFILE", "");
        if (!home.length) throw new Exception("Set TINYSTAN or CompileOptions.root; USERPROFILE is unavailable.");
    }
    else auto home = "~".expandTilde;
    return buildPath(home, ".tinystan", "tinystan-0.3.3");
}

version (Windows) unittest
{
    auto saved = environment.get("TINYSTAN", null);
    scope(exit) if (saved is null) environment.remove("TINYSTAN"); else environment["TINYSTAN"] = saved;
    environment.remove("TINYSTAN");
    assert(installationRoot("") == buildPath(environment["USERPROFILE"], ".tinystan", "tinystan-0.3.3"));
    environment["TINYSTAN"] = "from-env";
    assert(installationRoot("") == "from-env");
    assert(installationRoot("explicit") == "explicit");
}

/**
Compile a Stan model beside its source. Models with possible includes are retranslated on each call.
Params:
    source = Readable .stan source file.
    options = Installation and build settings.
Returns: Absolute path to the model-specific shared library.
Throws: Exception on invalid paths, an unsupported installation or build failure.
*/
string compileModel(string source, CompileOptions options = CompileOptions.init)
{
    source = source.absolutePath.buildNormalizedPath;
    if (source.extension != ".stan" || !source.exists || !source.isFile)
        throw new Exception("Expected a readable .stan file: " ~ source);
    auto root = installationRoot(options.root).absolutePath;
    auto header = buildPath(root, "src", "version.hpp");
    if (!buildPath(root, "Makefile").exists || !header.exists)
        throw new Exception("TinyStan 0.3.3 installation missing at " ~ root ~ ". Set TINYSTAN or CompileOptions.root.");
    auto lines = header.readText.splitLines;
    foreach (line; ["#define TINYSTAN_MAJOR 0", "#define TINYSTAN_MINOR 3", "#define TINYSTAN_PATCH 3"])
        if (!lines.canFind(line)) throw new Exception("Expected TinyStan 0.3.3 at " ~ root);
    auto output = source.stripExtension ~ "_model.so";
    auto make = options.make.length ? options.make : environment.get("MAKE", "make");
    auto args = [make] ~ options.makeArgs;
    if (options.stancArgs.length)
        args ~= "STANCFLAGS=" ~ (["--include-paths=."] ~ options.stancArgs).join(" ");
    version (Windows) source = source.replace("\\", "/");
    // The pinned Makefile tracks the source but omits transitive Stan includes.
    // A '#' also matches comments, where an occasional extra rebuild is harmless.
    if (source.readText.canFind('#'))
    {
        args ~= "--what-if=" ~ source;
        // Add the original default to the translation rule while inheriting
        // make/local flags (including additional include directories).
        if (!options.stancArgs.length)
            args ~= "--eval=%.hpp: STANCFLAGS += --include-paths=.";
    }
    args ~= source.stripExtension ~ "_model.so";
    auto run = execute(args, null, Config.none, size_t.max, root);
    if (run.status || !output.exists)
    {
        import std.format : format;
        throw new Exception(format("TinyStan build in %s exited %s\n%(%s %)\n%s", root, run.status, args, run.output));
    }
    return output;
}
