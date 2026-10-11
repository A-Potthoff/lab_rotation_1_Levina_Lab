# Scientific Annotation Template

### [Concept/Equation Name]
**Context:**
Why is this implemented this way?

**Derivation:**
$$

$$

**Source:**
- Source: [Citation]

---
### Back-Reference Selection (Delete what you don't use)

**Option A: Native Path (Cleanest, but fragile to file movement)**
Use when linking to a specific function name or static file.
[Return to implementation](../../src/simulation/biophysics.py)

**Option B: The URI Search Tag (Most robust against code moving lines)**
Requires a matching comment in the code like: `# [REF-UNIQUE-TAG] docs/annotations/file.md`
[Return to Code](vscode://search?query=REF-UNIQUE-TAG)

**Option C: Comment Anchors Extension (For global dashboards)**
Requires the VS Code 'Comment Anchors' extension. `# MATH: Equation 1`
