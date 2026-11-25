# DStan development

- Keep the public entry point at `import dstan`. Put dynamic symbols and foreign ownership details in private implementation code.
- Use the shared D compiler inventory, verify the selected executable, and pass its full path to DUB. Compare Windows and WSL using matching releases. Keep generated models for each platform in separate directories.
- Run `dub test` without a TinyStan installation. Use the integration command in the README for changes to compilation, loader behavior, ownership, sizing or sampling. Keep Ddoc examples and unit tests beside the declarations they exercise.
- The supported ABI is TinyStan 0.3.3. Update both upstream headers and their notice together, compare them to the release, and validate the C probe and real models before expanding compatibility.
- Document pointer ownership, retention and bounds before widening a trusted boundary. Preserve the upstream draw order. Keep Mir optional.
- Record verified platforms in the README and architecture notes. Keep future features in the roadmap. Original-code licensing awaits maintainer approval.
