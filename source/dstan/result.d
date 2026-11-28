/// Shape and storage of inference results in TinyStan's native order.
module dstan.result;

/// Final L-BFGS output, usable after closing the model.
struct OptimizationResult
{
    double[] values;           /// One D-owned row, in the same order as names.
    immutable(string)[] names; /// lp__, converged__, then flattened model output names.
    bool jacobian;             /// Parameter-transform adjustment used to optimize the objective.
}

/// D-owned draws and final per-chain tuning values from adaptive NUTS.
struct AdaptationResult
{
    Draws draws;            /// Saved warmup and posterior draws, as selected by SampleOptions.
    double[] stepSizes;     /// Final nominal step size for each chain, in chain order.
    double[] inverseMetric; /// One block per chain, in initialInverseMetric layout. Empty for the unit metric.
}

/// Sampling results own their flat D storage and copied column metadata.
struct Draws
{
    double[] values;             /// Contiguous chain, draw, column storage.
    immutable(string)[] names;   /// Diagnostic columns followed by flattened Stan outputs.
    size_t[3] shape;             /// Chains, saved rows per chain (including saved warmup), columns.
    size_t savedWarmup;          /// Number of leading warmup rows per chain, zero by default.

    /**
    Params:
        chain = Zero-based chain index.
        draw = Zero-based saved draw index.
        column = Zero-based output column.
    Returns: The value at the requested indices.
    Throws: Exception for out-of-range indices.
    */
    double opIndex(size_t chain, size_t draw, size_t column) const @safe
    {
        if (chain >= shape[0] || draw >= shape[1] || column >= shape[2])
            throw new Exception("Draw index outside result dimensions");
        return values[(chain * shape[1] + draw) * shape[2] + column];
    }
    ///
    unittest
    {
        auto draws = Draws([0.1, 2.0, 0.2, 3.0], ["lp__", "theta"], [1, 2, 2]);
        assert(draws[0, 1, 1] == 3.0);
    }
}

package size_t checkedCount(int chains, int draws, size_t columns) @safe
{
    if (chains <= 0 || draws <= 0 || columns < 7 || columns > int.max / draws)
        throw new Exception("Draw buffer exceeds TinyStan signed-int limits or has invalid dimensions");
    size_t perChain = columns * draws;
    if (perChain > size_t.max / double.sizeof / chains)
        throw new Exception("Draw buffer byte count overflows size_t");
    return perChain * chains;
}

unittest
{
    import std.exception : assertThrown;
    assert(checkedCount(2, 4, 8) == 64);
    assertThrown(checkedCount(0, 4, 8));
    assertThrown(checkedCount(1, int.max / 8 + 1, 8));
    assertThrown(checkedCount(1, 0, 8));
    assertThrown(checkedCount(1, 1, size_t.max));
    assertThrown(checkedCount(int.max, 1, int.max));
    static if (size_t.sizeof == 8)
        assert(checkedCount(2, 1, int.max) == 2 * cast(size_t)int.max);
}
