# Working preferences for this repository

---
(instructions fetched from https://github.com/multica-ai/andrej-karpathy-skills/tree/main)

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

---
# personal additions

## Code style
- Readability first. Use descriptive names, e.g. `counter` not `cntr`, `rhs` for a right-hand side. Avoid one-letter variable names unless it is absolutely clear, or standard notation (eg i, j for iterations)
- Keyword arguments where they clarify meaning, e.g. `np.random.default_rng(seed=0)` instead of `default_rng(0)`.
- Don't hard-wrap prose in comments and docstrings: one line per sentence or paragraph is fine.
- Comments explain the *why* and the formula being implemented. Don't add trivial comments.
- Small inconsistencies (bad names, truncated comments, dead code) are simply fixed, with a short comment where the reason isn't obvious. They don't need entries in a flag list.
- use explicit mappings where possible without overhead of code/compute. eg. for plotting, I prefer a {"state1": "blue", "state2":"red"} dict over an abstract index that is hard to read.
- write compact code, in this sense: adapt functions instead of writing new ones where appropriate. de-duplicate logic when the context is similar, and you can do it without overhead.
- in a SHORT comment, supplement the resulting structure of a command, if not trivial. eg.
  - \# Commment: Wake = ndarray(ndarray(float))
  - WAKE, NREM, REM = ( states.loc[states["state"] == s, ["start_s", "end_s"]].to_numpy() for s in ("WAKE", "NREM", "REM"))
- or anything similar that is short and helps parsing code more quickly

## Notebooks
- Load `jupyter_black` in the first code cell (`%load_ext jupyter_black`), so cells are formatted consistently.
- Do **not** save figures or results by default. Include the `savefig` / `np.save` calls, but commented out. Only outputs the user has decided to keep belong in `results/figures/` or `results/objects/`.
- In analyses, print fairly verbose statements after cells (shapes, dtypes, value ranges, units, what a result means), so the user gets a feeling for the data and the type of the answers.
- Use the kernel `levina_rotation_1` ("Python (Levina_rotation_1)").
- put insights in comments, not so much in fully formatted print statements. do both, but keep cell outputs rather data-centered.

## Structure
- general functions are good and useful, but don't export every function definition to source. Example: if we deal with the data in only ever one notebook, do not write functions for reading in the data into source. Write it in the notebook instead. otherwise, we clutter the source folder with highly-specific underused functions.
- use global path variables sparsely. results, data_dirs, etc are sensible. but not every single data file needs a path in src/config/paths. use it if necessary, don't clutter. you can define paths in notebooks if needed.
- the same goes for variables. if we only need a list once for iteration, a "for states in ["WAKE", "REM", "NREM"]" is fine and we don't need a states_list variable for this. use it, if it gets reused, though!