# CLAUDE.md

## About this project

`wc_cv` is the research and documentation repository for **combra** — computer-vision
tools for analysis of WC-Co (tungsten-carbide / cobalt) composite-alloy microstructure
SEM images: contour/angle extraction, MVEE beam fitting, fractal dimension, crack
graphs, and distribution metrics.

- `docs/` is a **Sphinx** site (MyST markdown) that documents the `combra` library API.
  It is published to GitHub Pages on every push to `main`. Build it locally with:
  ```bash
  python -m sphinx -b html docs public
  ```
  Doc pages live in `docs/api/*.md` (one per combra submodule), `docs/get_started.md`,
  and `docs/examples/*.md`. When the combra library changes, keep these pages in sync.
- `combra` itself is maintained in a **separate repo** (`dkagramanyan/combra`, wired in
  here as the `combra/` git submodule). Run its test suite from that repo with `pytest`.
- The top-level notebooks (`co_angles/`, `crack_graph/`, `ml/`, `poliamid/`) are
  exploratory analyses that exercise the documented combra API.

---

Behavioral guidelines to reduce common LLM coding mistakes, derived from
[Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876)
on LLM coding pitfalls. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
