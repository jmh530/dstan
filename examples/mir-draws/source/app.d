import columns : posteriorColumn;
import dstan;
import std.exception : enforce;

void main(string[] args)
{
    import std.stdio : writeln;

    enforce(args.length >= 2, "Pass tests/models/normal.stan and optional Windows DLL directories");
    auto model = Model(compileModel(args[1]), `{"mu": 2.5}`, 123, args[2 .. $]);
    SampleOptions options;
    options.seed = 321;
    options.chains = 2;
    options.draws = 4000;
    options.saveWarmup = true;
    auto fit = model.sample(options);

    foreach (chain; 0 .. fit.shape[0])
    {
        auto theta = posteriorColumn(fit, chain, "theta");
        double sum = 0;
        foreach (value; theta)
            sum += value;
        writeln("Chain ", chain + 1, " posterior mean of theta: ", sum / theta.length);
    }
}
