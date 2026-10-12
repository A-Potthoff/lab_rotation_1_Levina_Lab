# Working preferences for this repository

## Code style
- Readability first. Use descriptive names, e.g. `counter` not `cntr`, `rhs` not `b` for a right-hand side.
- Keyword arguments where they clarify meaning, e.g. `np.random.default_rng(seed=0)` instead of `default_rng(0)`.
- Don't hard-wrap prose in comments and docstrings: one line per sentence or paragraph is fine.
- Comments explain the *why* and the formula being implemented. Don't add trivial comments.
- Python code is formatted with black
- Small inconsistencies (bad names, truncated comments, dead code) are simply fixed, with a short comment where the reason isn't obvious. They don't need entries in a flag list.

## Notebooks
- Load `jupyter_black` in the first code cell (`%load_ext jupyter_black`), so cells are formatted consistently.
- Do **not** save figures or results by default. Include the `savefig` / `np.save` calls, but commented out. Only outputs the user has decided to keep belong in `reports/figures/` or `results/objects/`.
- In analyses, print fairly verbose statements after cells (shapes, dtypes, value ranges, units, what a result means), so the user gets a feeling for the data and the type of the answers.
- Use the kernel `levina_rotation_1` ("Python (Levina_rotation_1)").