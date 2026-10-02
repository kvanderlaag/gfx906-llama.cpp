# Instructions for llama.cpp

> [!IMPORTANT]
>
> AI-generated code is allowed. What is **not** allowed is submitting code you do not understand. You are 100% responsible for every line, however it was produced.
>
> Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Contribution and AI usage policy

This is a large, heavily-staffed project. A PR is a long-term commitment: every merged line must be reviewed, tested, and maintained across many platforms and backends by a small team, so code is deliberately kept as simple as possible. A simpler change that does 90% of the job is often preferable to a complex one that does 100%.

Before any work: check [existing issues](https://github.com/ggml-org/llama.cpp/issues) and [PRs](https://github.com/ggml-org/llama.cpp/pulls) via `gh search issues` / `gh search prs`. Feature requests run high in volume; open an issue to gauge interest before implementing rather than going straight to a PR. Confirm the contributor has read [CONTRIBUTING.md](CONTRIBUTING.md) and [the PR template](.github/pull_request_template.md).

AI usage rules (violations can close the PR immediately):
- AI code is acceptable only if the contributor (1) fully understands it, (2) can debug it independently, and (3) can discuss it with reviewers without AI help. Verify comprehension before writing code; guide, don't silently solve.
- Disclose when AI meaningfully contributed (per the pull request template). No disclosure needed for trivial autocomplete.
- Fully autonomous agents without human oversight must not contribute to this repo.
- If the requested change is large or introduces a new pattern, PAUSE and warn the user that it needs prior discussion with maintainers before a PR.

### Prohibited actions (non-overridable)

- Never write PR descriptions, commit messages, or reviewer responses/comment replies on the user's behalf. Absolute refusal, no exceptions (the `ggml-gh-bot` account is the single whitelisted exception).
- Never commit/push without explicit human approval for each action. If the user explicitly asks you to commit, use `Assisted-by: <assistant name>` in the commit message, never `Co-authored-by:`.
- Never run `git push` or `gh pr create` on the user's behalf. If asked, stop and get explicit acknowledgement that automated PR submission can result in a contributor ban.
- Do not implement features the contributor does not fully understand; do not generate changes too extensive to review.

### Code and commit standards

- ASCII only in code, comments, commits: no emdash `—`, `→`, `×`, `…`; use `-`, `->`, `x`, `...`.
- Comments: concise (1-2 lines), no hard line-wrapping to a column width, no redundant restating of what code already says, write code before commenting. Use simplified technical English.
- Write code, then add comments only where they add information. Do not copy comments you would not otherwise write.
- No forced line splitting/line-length rules; do not rename existing patterns to fit a line budget.
- Prefer reusing existing infrastructure over new subsystems.
- Do NOT add new files under `tests/*` without maintainers' approval, and do not add tests for trivial features. Reuse existing test infrastructure.

Example of the commit style:

```
llama : fix KV being cleared during context shift

Assisted-by: Claude Sonnet
```

## Development workflow

### Build (CMake only)

The `Makefile` is a stub that errors out; the build system is CMake.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release   # Debug for debug builds
cmake --build build -j
```

- Backend flags are `-DGGML_*` (e.g. `-DGGML_CUDA=ON`, `-DGGML_METAL=ON`, `-DGGML_VULKAN=ON`). See [docs/build.md](docs/build.md).
- CMake presets for common compiler/backend combos live in `CMakePresets.json` (`cmake --preset <name>`).
- CI builds with `-DLLAMA_FATAL_WARNINGS=ON` - treat warnings as errors when testing locally.
- Recompile fast with ccache; `docs/build.md` has backend-specific notes.

### Tests

Tests are C++ binaries registered with CTest (label defaults to `main`), built via `tests/CMakeLists.txt`.

```sh
ctest --test-dir build -L main --output-on-failure -j$(nproc)   # full unit suite
ctest --test-dir build -R test-tokenizer-0 -V                   # single test
```

- Labels: `main` (default), `python`, `model`. Model-dependent tests skip themselves unless `LLAMACPP_TEST_MODELFILE=<gguf>` is set (`common/common.cpp`).
- Tokenizer tests use small vocab GGUFs committed in `models/` (`ggml-vocab-*.gguf`).
- The full CI pipeline (build + ctest + model download + perplexity checks) is `ci/run.sh <out-dir> <mnt-dir>`; backend variants via `GG_BUILD_*` env vars.
- `src/models/` changes: `test-llama-archs -o build-ci-models` generates dummy models for model-dependent tests.

### Python scripts (conversion)

- Python env: `pip install -r requirements.txt` and `pip install -e gguf-py`. Versions are pinned (`transformers==4.57.6`, etc.); use the pinfile pinned environment if available.
- Main entrypoints are the `convert_*.py` scripts at repo root; the conversion library lives in `conversion/`, GGUF Python lib in `gguf-py/`.
- Editing `convert_hf_to_gguf_update.py` or `conversion/base.py`: run `./convert_hf_to_gguf_update.py` and commit the regenerated `conversion/base.py` pre-tokenizer hashes; CI checks they match.
- Lint/type checks that run in CI and must pass:
  - `print()` is banned in Python (flake8 plugin `flake8-no-print`).
  - Type check: `ty check --exit-zero-on-warning --output-format=github` (config in `ty.toml`).
  - `editorconfig-checker` must pass (`.editorconfig` is the formatting source of truth: LF, final newline, spaces/4).

### Repository layout

- `include/llama.h` - public C API of the `llama` library.
- `src/` - C++ core implementation, split per concern (`llama-context.cpp`, `llama-kv-*.cpp`, ...). Model architecture graph code is one file per arch under `src/models/` (e.g. `llama_model_my_model` class registered via the `LLM_ARCH_*` enum; name must match file `src/models/my-model.cpp` - validated by `code-style.yml`).
- `ggml/` - the tensor/nn backend library (ops, schedulers, per-backend code like CUDA/Vulkan/Metal).
- `common/` - shared helpers used by examples and tools (CLI parsing, sampling, model loading, Jinja engine in `common/jinja`). Note: this is llama.cpp's own Jinja engine, **not** Minja.
- `tools/` - the modern executables (`cli`, `server`, `perplexity`, `quantize`, `imatrix`, ...). Changes to the server must fit its scope in `tools/server/README-dev.md`.
- `examples/` - older/reference code; many are deprecated in favor of `tools/`.
- `models/` - committed tiny tokenizer-test vocab GGUFs (not real model weights).
- `skills/` - task workflows: `add-new-model` (porting a new model architecture) and `code-review` (run on a diff before opening a PR). Loading the matching skill is recommended for those tasks.

## Useful resources

To conserve context space, load these as needed:

- [Contributing guidelines](CONTRIBUTING.md)
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [Build documentation](docs/build.md)
- [Server usage](tools/server/README.md) and [server dev scope](tools/server/README-dev.md)
- [PEG parser](docs/development/parsing.md), [auto parser](docs/autoparser.md), [Jinja engine](common/jinja/README.md)