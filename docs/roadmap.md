# Roadmap

The first milestone covers compilation, dynamic loading, JSON model construction and NUTS across one or more chains into D-owned or caller-provided storage.

1. Qualify macOS and additional D compiler versions. Test GDC's declaration-import options before claiming support. Compare headers and repeat ABI, layout, lifetime and statistical tests before adding a TinyStan release.
2. Extend interrupt qualification to macOS. Extend result-size checks for each algorithm. Evaluate user callbacks separately, including D runtime thread registration and exception containment.
3. The optional Mir example covers named posterior columns. Integration tests cover flattened vectors, matrices and arrays of vectors. Consider a public adapter as recurring use cases emerge and evaluate Mir JSON adapters. Reuse mir-stat for descriptive summaries. Investigate rank-normalized R-hat and bulk/tail ESS with Mir maintainers.
4. L-BFGS optimization exposes the final named row, and Laplace sampling accepts that row as its mode. Evaluate additional optimizers, tuning controls, optional Hessian output and Pathfinder. Evaluate standalone generated quantities and explicit cancellation with TinyStan upstream. Checked size arithmetic and clarification of the Laplace mode buffer length in TinyStan would benefit all clients.
5. Measure cold/warm compilation, loading, initialization, sampling, output access, allocations and peak memory independently. Compare repeated small fits and large outputs with the same-binary Python interface before comparing equivalent CmdStan runs.

Consider automatic installation and caching after the supported build combinations are established. Cache keys need OS, architecture, upstream revisions, C++ compiler and flags. Model keys also need included Stan source, stanc version and options. Keep D build keys independent. Package publication and final licensing require maintainer review.
