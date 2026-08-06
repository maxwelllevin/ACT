# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Dev/test environment

`requirements.txt` pinned to modern versions can fail on Python 3.9 (pip resolves
xarray <2024.9 which lacks `xr.coders`, breaking `act/io/arm.py`). Use the conda
env from `environment.yml` (`conda env create -f environment.yml`, Python 3.11)
for running the test suite; it also needs `pytest-cov` and, if working on
`act.transform`, `numba` (already listed in both `environment.yml` and
`requirements.txt`).

## `lazy_loader.attach` pitfall: function name == defining module name

If a subpackage's `__init__.py` uses `lazy_loader.attach()` and a public function
shares its name with the file that defines it (e.g. `act/transform/bin_average.py`
defining a function also called `bin_average`), do NOT list that module in
`submodules` and rely on `submod_attrs` alone. `lazy_loader.attach`'s lazy
resolution caches whichever was imported last: if any code elsewhere does a bare
`import act.transform.bin_average` after the package-level attribute was already
resolved to the function, `act.transform.bin_average` silently becomes the
*module* object instead of the function, and calling it raises
`TypeError: 'module' object is not callable`. Fix: import that name eagerly at
the top of `__init__.py` (`from .bin_average import bin_average`) instead of
routing it through `lazy.attach`'s `submod_attrs`. See `act/transform/__init__.py`.

Doing so has a documentation consequence: `lazy.attach` derives its `__dir__` only
from the names it was asked to load lazily, so eagerly imported names drop out of
`dir()`. `sphinx.ext.autosummary` builds each module's API page from `dir()`, so
such a name is silently omitted from the generated docs while still importing and
working normally — nothing fails, the page is just missing entries. Any subpackage
that overrides `__all__` after `lazy.attach` should define `__dir__` to match it;
`act/transform/__init__.py` does, and `tests/transform/test_api.py::TestPublicSurface`
guards it.

## `read_arm_netcdf` leaves CF bounds variables as cftime objects

`act.io.arm.read_arm_netcdf` decodes with `use_cftime=True` and then converts only
`time`/`time_offset` back to `datetime64` (see `act/io/arm.py`). Any other time-like
variable — notably a CF `time_bounds` — stays a dtype-`object` array of
`cftime.DatetimeGregorian`. So `ds['time']` is `datetime64[ns]` while
`ds['time_bounds']` is not, and `np.asarray(bounds, dtype=float)` on it raises
`TypeError`. Reading the same file with plain `xr.open_dataset` yields `datetime64`
bounds and hides the problem, so exercise any bounds-consuming code through ACT's
own reader. `act/transform/driver.py:_to_numeric` is the normalization helper.

## Pre-existing `pytest tests/` failures

Roughly 7 failures are unrelated to any local change: network-dependent
`tests/discovery/` tests (403 through a proxy) plus upstream API drift in
`test_ameriflux`, `test_bsrn_limits_test`, `test_sonde`, and `test_convert_units`.
Confirm a failure reproduces on a clean checkout before investigating it. Some
`tests/discovery/` and `tests/retrievals/` tests also write download artifacts
(`data/`, `Boulder_CO_surfrad/`) into the repo root; delete them before committing.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
