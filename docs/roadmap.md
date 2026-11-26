# Roadmap

The first milestone covers compilation, dynamic loading, JSON model construction and NUTS across one or more chains into D-owned or caller-provided storage.

1. Qualify macOS and additional D compiler versions. Test GDC's declaration-import options before claiming support. Compare headers and repeat ABI, layout, lifetime and statistical tests before adding a TinyStan release.
2. Test process interrupt behavior during parallel sampling. Support caller-supplied inverse metrics and retrieval of adapted metrics, with checked per-chain buffer shapes. Extend result-size checks for each algorithm. Evaluate user callbacks separately, including D runtime thread registration and exception containment.
3. Add optional Mir views and verified JSON adapters as demand develops. Verify vector, matrix and array ordering against upstream fixtures. Reuse mir-stat for descriptive summaries. Investigate rank-normalized R-hat and bulk/tail ESS with Mir maintainers.
4. Expose optimization, Pathfinder and Laplace using their individual contracts. Evaluate standalone generated quantities and explicit cancellation with TinyStan upstream. Checked size arithmetic in TinyStan would benefit all clients.
5. Measure cold/warm compilation, loading, initialization, sampling, output access, allocations and peak memory independently. Compare repeated small fits and large outputs with the same-binary Python interface before comparing equivalent CmdStan runs.

Consider automatic installation and caching after the supported build combinations are established. Cache keys need OS, architecture, upstream revisions, C++ compiler and flags. Model keys also need included Stan source, stanc version and options. Keep D build keys independent. Package publication and final licensing require maintainer review.
