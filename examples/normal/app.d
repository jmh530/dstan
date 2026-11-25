import dstan;
import std.stdio : writeln;

void main(string[] args)
{
    import std.exception : enforce;
    enforce(args.length >= 2, "Pass tests/models/normal.stan and optional Windows DLL directories");
    auto model = Model(compileModel(args[1]), `{"mu": 2.5}`, 123, args[2 .. $]);
    SampleOptions options;
    options.seed = 321;
    options.draws = 4000;
    auto result = model.sample(options);
    writeln(result.shape, " ", result.names);
    double sum = 0;
    foreach (i; 0 .. options.draws) sum += result[0, i, 7];
    writeln("Posterior mean of theta: ", sum / options.draws);
}
