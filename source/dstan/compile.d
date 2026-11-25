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
    string[] stancArgs;     /// Trusted stanc flags interpreted by the upstream Make recipe.
}

/**
Compile a Stan model beside its source. GNU Make determines which targets are stale.
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
    auto root = options.root;
    if (!root.length) root = environment.get("TINYSTAN", "");
    if (!root.length) root = buildPath("~".expandTilde, ".tinystan", "tinystan-0.3.3");
    root = root.absolutePath;
    auto header = buildPath(root, "src", "version.hpp");
    if (!buildPath(root, "Makefile").exists || !header.exists)
        throw new Exception("TinyStan 0.3.3 installation missing at " ~ root ~ ". Set TINYSTAN or CompileOptions.root.");
    auto lines = header.readText.splitLines;
    foreach (line; ["#define TINYSTAN_MAJOR 0", "#define TINYSTAN_MINOR 3", "#define TINYSTAN_PATCH 3"])
        if (!lines.canFind(line)) throw new Exception("Expected TinyStan 0.3.3 at " ~ root);
    auto output = source.stripExtension ~ "_model.so";
    auto make = options.make.length ? options.make : environment.get("MAKE", "make");
    auto target = output;
    version (Windows) target = target.replace("\\", "/");
    auto args = [make] ~ options.makeArgs
        ~ ["STANCFLAGS=" ~ (["--include-paths=."] ~ options.stancArgs).join(" "), target];
    auto run = execute(args, null, Config.none, size_t.max, root);
    if (run.status || !output.exists)
    {
        import std.format : format;
        throw new Exception(format("TinyStan build in %s exited %s\n%(%s %)\n%s", root, run.status, args, run.output));
    }
    return output;
}
