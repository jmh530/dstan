module columns;

import dstan;
import std.exception : enforce;

/**
Params:
    fit = Sampling result whose storage backs the view.
    chain = Zero-based chain index within the result.
    name = Exact flattened output or diagnostic column name.
Returns: A one-dimensional strided Mir slice of the chain's posterior values.
    Writing through the slice updates the corresponding elements of fit.values.
Throws: Exception if the name is absent.
*/
auto posteriorColumn(Draws fit, size_t chain, string name)
{
    import mir.ndslice : sliced;
    import std.algorithm.searching : countUntil;

    auto column = fit.names.countUntil(name);
    enforce(column >= 0, "Unknown draw column: " ~ name);
    auto draws = fit.values.sliced(fit.shape);
    return draws[chain, fit.savedWarmup .. $, cast(size_t) column];
}

unittest
{
    auto fit = Draws([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        ["theta", "lp__"], [2, 3, 2], 1);
    foreach (chain; 0 .. fit.shape[0])
    {
        auto theta = posteriorColumn(fit, chain, "theta");
        assert(theta.length == 2);
        foreach (i; 0 .. theta.length)
            assert(theta[i] == fit[chain, fit.savedWarmup + i, 0]);
    }
    auto lp = posteriorColumn(fit, 1, "lp__");
    lp[1] = -42;
    assert(fit[1, 2, 1] == -42);
}
