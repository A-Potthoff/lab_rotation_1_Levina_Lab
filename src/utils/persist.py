"""Save and load results to disk, so long computations only have to run once.

You pass the objects as keyword arguments. Their names become the names you get back
when loading. The file name is chosen for you, <tag>__<YYYY-MM-DD_HHMMSS>.pkl, so old
files are never overwritten. The default folder is data/interim/.

Examples
--------
    from src.utils.persist import save, load, list_saves, prune

    # Save after a long run, load in a separate plotting cell
    path = save("pif_sweep", results=results, distances=distances, params=params)
    data = load(path)
    data.results                  # same as data["results"]

    # After a kernel restart (the variable `path` is gone): newest file with this tag
    data = load("pif_sweep")

    # Without a tag the name is built from the object names: "results-distances__..."
    save(results=results, distances=distances)
    data = load()                 # newest file in the folder, whatever its tag

    # Other folders (see FOLDERS below) or any path; optional gzip compression
    save("spike_counts", counts=counts, folder="processed", compress=True)
    data = load("spike_counts", folder="processed")

    # Debug runs pile up: keep only the 3 newest files of this tag
    save("pif_sweep_debug", results=results, keep=3)

    # Checkpoint inside a long loop, so a crash at 4 am loses at most one step
    for i, d in enumerate(distances):
        ...
        save("pif_sweep_partial", results=results, folder="checkpoints", keep=1)

    # Housekeeping and provenance
    list_saves()                  # files in the folder, with sizes
    prune("pif_sweep", keep=2)    # delete all but the 2 newest of this tag
    data.meta                     # tag, save time, git commit, file path
    globals().update(data)        # restore the variables under their original names

Reserved names: tag, folder, compress and keep are options of save(), and meta is used
for data.meta, so objects cannot be saved under these names.

Anything picklable works: numpy arrays, pandas tables, nested lists/dicts, NamedTuples
and other custom classes. A custom class must be importable under the same module path
when loading, so keep result classes in src/ rather than defining them in a notebook.
Only load files you created yourself (pickle can execute code).
"""

from __future__ import annotations

import gzip
import pickle
import re
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

from src.config.paths import DATA_INTERIM, DATA_PROCESSED, PROJECT_ROOT, RESULTS_OBJECTS

# Short names you can pass as `folder=...`
FOLDERS = {
    "interim": DATA_INTERIM,  # default: intermediate results such as simulation sweeps
    "processed": DATA_PROCESSED,  # final data sets
    "checkpoints": DATA_INTERIM / "checkpoints",  # partial results of long runs
    "results": PROJECT_ROOT / "results",
    "objects": RESULTS_OBJECTS,  # results/objects
}

_OPTIONS = ("tag", "folder", "compress", "keep")  # names taken by save's own options

# File names look like <tag>__<YYYY-MM-DD_HHMMSS>[_n].pkl[.gz]
_NAME = re.compile(r"(?P<tag>.+)__(?P<stamp>\d{4}-\d\d-\d\d_\d{6}(?:_\d+)?)\.pkl(?:\.gz)?")


class Bundle(dict):
    """A dict whose entries can also be read as attributes: data.results is data["results"]."""

    def __getattr__(self, key):
        # Python calls this only if `key` is not a normal attribute: then look in the dict.
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __dir__(self):
        """Lets Tab completion in Jupyter show the saved names."""
        return [*super().__dir__(), *self.keys()]

    def __repr__(self):
        return f"Bundle(\n{_summary(self)}\n)"


def save(tag=None, *, folder="interim", compress=False, keep=None, **objects) -> str:
    """Pickle all keyword arguments into one new file and return the file's path.

    tag:      label at the start of the file name. Default: the object names joined,
              e.g. "results-distances".
    folder:   "interim" (default), "processed", "checkpoints", "results", "objects"
              (see FOLDERS), or any path. Created if missing.
    compress: gzip the file. Helps for integer or sparse data, hardly for noisy floats.
    keep:     afterwards keep only the newest `keep` files of this tag. None: delete nothing.
    """
    _check_options(tag, folder, compress, keep, objects)
    if tag in FOLDERS:
        warnings.warn(f'tag "{tag}" is a folder name. Did you mean folder="{tag}"?', stacklevel=2)
    target = _folder(folder)
    target.mkdir(parents=True, exist_ok=True)
    tag = _clean(tag or "-".join(objects))
    now = datetime.now()
    stamp = now.strftime("%Y-%m-%d_%H%M%S")
    suffix = ".pkl.gz" if compress else ".pkl"

    path, n = target / f"{tag}__{stamp}{suffix}", 1
    while path.exists():  # several saves within the same second
        n += 1
        path = target / f"{tag}__{stamp}_{n}{suffix}"

    # What goes into the file: a marker ("written by save"), info about this save, your objects
    meta = {"tag": tag, "saved": now.strftime("%Y-%m-%d %H:%M:%S"), "git": _git_commit()}
    payload = {"__persist__": 1, "meta": meta, "objects": dict(objects)}

    tmp = path.with_name(path.name + ".part")  # write, then rename: never a half-written file
    with gzip.open(tmp, "wb", compresslevel=6) if compress else open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)

    git = f", git {meta['git']}" if meta["git"] else ""
    shown = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
    where = "" if folder == "interim" else f', folder="{folder}"'
    print(f"Saved {shown} ({_size(path)}{git})\n{_summary(objects)}")
    print(f'→ reload: load("{path.name}"{where})  or newest of this tag: load("{tag}"{where})')
    if keep is not None:
        prune(tag, keep=keep, folder=folder)
    return str(path)


def load(name=None, *, folder="interim"):
    """Load a file written by save() and return a Bundle of its objects.

    name:   a path or file name (as returned or printed by save), or a tag (loads the
            newest file with that tag), or None (loads the newest file in `folder`).
    folder: where to look if `name` is not a full path; same choices as in save().

    The result also has `.meta` with the tag, save time, git commit and file path.
    Plain pickles not written by save() are returned unchanged.
    """
    path = _resolve(name, _folder(folder))
    with gzip.open(path, "rb") if path.name.endswith(".gz") else open(path, "rb") as f:
        payload = pickle.load(f)
    if not (isinstance(payload, dict) and payload.get("__persist__")):
        print(f"Loaded {path.name} (plain pickle, {_size(path)})")
        return payload

    data = Bundle(payload["objects"])  # your objects ...
    data.meta = Bundle(payload["meta"], path=str(path))  # ... plus the info about the save
    git = f", git {data.meta['git']}" if data.meta.get("git") else ""
    print(f"Loaded {path.name} ({_size(path)}{git})\n{_summary(data)}")
    return data


def list_saves(tag=None, *, folder="interim") -> None:
    """Print the saved files in `folder` (oldest first), optionally only those of one tag."""
    files = _find(_folder(folder), tag)
    for p in files:
        print(f"{p.name:<70} {_size(p):>9}")
    if not files:
        print(f"No saves{f' with tag {tag!r}' if tag else ''} in {_folder(folder)}")


def prune(tag, keep=3, *, folder="interim") -> None:
    """Delete all but the newest `keep` files of this tag."""
    if keep < 1:
        raise ValueError("keep must be at least 1")
    for p in _find(_folder(folder), tag)[:-keep]:
        p.unlink()
        print(f"Deleted old {p.name}")


# ---------------------------------------------------------------- helpers


def _check_options(tag, folder, compress, keep, objects) -> None:
    """Stop with a clear message if an object was (probably) given a reserved name.

    Python hands tag=..., folder=..., compress=... and keep=... to the options of save(),
    so an object with one of these names never reaches **objects and would be missing from
    the file without notice. This is only detectable when the value has the wrong type
    for that option (a variable `keep = 3` would still be taken as the keep option).
    """
    valid = {
        "tag": tag is None or isinstance(tag, str),
        "folder": isinstance(folder, (str, Path)),
        "compress": isinstance(compress, bool),
        "keep": keep is None or (isinstance(keep, int) and not isinstance(keep, bool)),
    }
    for name, ok in valid.items():
        if not ok:
            raise TypeError(
                f"save() got an unexpected value for its option '{name}'. If you meant to save "
                f"an object called '{name}', rename it, e.g. save({name}_data={name}). "
                f"Reserved names: {', '.join(_OPTIONS)}, meta."
            )
    if "meta" in objects:
        raise ValueError("'meta' is reserved for data.meta; save the object under another name.")
    if not objects:
        raise ValueError("Nothing to save: pass objects as keywords, e.g. save(results=results).")


def _folder(folder) -> Path:
    """Turn a short name like "interim" into its path (see FOLDERS).

    Path objects and strings containing a slash are used as paths directly. A plain word
    that is not in FOLDERS is an error, so a typo cannot create a stray folder.
    """
    if isinstance(folder, str) and "/" not in folder and "\\" not in folder:
        if folder not in FOLDERS:
            raise ValueError(f"Unknown folder {folder!r}. Use one of {list(FOLDERS)} or a path.")
        return FOLDERS[folder]
    return Path(folder)


def _clean(text) -> str:
    """Make a string safe for a file name: special characters become "_", max. 60 characters.

    save() and load() both use this, so a tag always finds its own files.
    """
    return re.sub(r"[^\w.-]+", "_", str(text))[:60].strip("_.-") or "data"


def _find(folder: Path, tag=None) -> list[Path]:
    """All files in `folder` named by save() (optionally one tag only), oldest first.

    The order comes from the time stamp in the file name, not from the file system.
    """
    if not folder.is_dir():
        return []
    tag = None if tag is None else _clean(tag)
    hits = []
    for p in folder.iterdir():
        m = _NAME.fullmatch(p.name)
        if m and (tag is None or m["tag"] == tag):
            hits.append((m["stamp"], p))
    return [p for _, p in sorted(hits)]


def _resolve(name, folder: Path) -> Path:
    """Find the file that load(name) means. Tried in this order:

    1. `name` is an existing file: a full path, or a file name inside `folder`.
    2. `name` is a tag: the newest file with that tag in `folder`.
    3. `name` is None: the newest file in `folder`, whatever its tag.
    """
    if name is not None:
        for p in (Path(name), folder / name):
            if p.is_file():
                return p
    files = _find(folder, name)
    if not files:
        what = f"file or tag {str(name)!r}" if name is not None else "saves"
        hint = f' Did you mean load(folder="{name}")?' if name in FOLDERS else ""
        raise FileNotFoundError(f"No {what} in {folder}.{hint} Run list_saves() to see what exists.")
    return files[-1]


def _describe(obj) -> str:
    """One-line description of an object for the printouts, e.g. "ndarray float64 (10, 2)"."""
    kind = type(obj).__name__
    shape = getattr(obj, "shape", ())
    if shape:  # numpy arrays, pandas tables, ...
        dtype = f" {obj.dtype}" if hasattr(obj, "dtype") else ""
        return f"{kind}{dtype} {tuple(shape)}"
    if isinstance(obj, (list, tuple, dict, set)) and not hasattr(obj, "_fields"):
        text = repr(obj) if len(obj) <= 8 else ""
        return text if 0 < len(text) <= 70 else f"{kind}, len {len(obj)}"
    if isinstance(obj, str):  # long strings (e.g. paths): cut the middle, keep start and end
        return repr(obj) if len(obj) <= 120 else repr(f"{obj[:40]} … {obj[-75:]}")
    text = repr(obj)
    return text if len(text) <= 70 else kind  # long reprs (e.g. a DbResult): type name only


def _summary(objects: dict) -> str:
    """One line per object, "  name : description", with the colons aligned."""
    width = max(map(len, objects), default=0)
    return "\n".join(f"  {k:<{width}} : {_describe(v)}" for k, v in objects.items())


def _size(path: Path) -> str:
    """File size as readable text, e.g. "8.3 kB"."""
    n = path.stat().st_size
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def _git_commit() -> str | None:
    """Short id of the current git commit (e.g. "28b68ee"), or None if git is unavailable.

    It tells you which version of the code produced a file. Changes that were not yet
    committed at save time are not part of that id.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None