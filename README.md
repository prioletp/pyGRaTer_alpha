# pyGrater

pyGrater is a Python package for debris disk modeling and radiative transfer of optically thin media.

## Installation

To install, run in the top directory (containing pyproject.toml):

```bash
pip install .
```

Or install in editable (developer) mode:

```bash
pip install -e .
```

---

## Data Setup

pyGrater requires an external data directory (optical properties, stellar catalogs, filters, etc.).

Download the data from:
https://osf.io/mqkyf/overview

Then configure the data path using one of the following options.

### Option 1: configure inside Python

```python
import pyGrater
pyGrater.set_data_path("/path/to/downloaded/data")
```

### Option 2: environment variable

```bash
export PYGRATER_DATA_PATH=/path/to/downloaded/data
```

### Option 3: command-line helper (only if installed via pip)

```bash
pygrater-setup --data-path /path/to/downloaded/data
```

Check current configuration:

```bash
pygrater-setup --show
```

Notes:

- The persistent config is stored in ~/.pygrater/config.json.
- If no data path is configured, pyGrater raises a FileNotFoundError with setup instructions.


## How to Add a New Star

To add a star, you must edit the star catalog text file in the data directory:

data/star_data/stars_main_properties.txt

### Important format rules

1. Keep the existing header columns unchanged.
2. Add one new row with whitespace-separated values.
3. Do not use spaces inside the star name token.
4. Use nan for unknown numeric values.

Expected columns are:

star dist temp rad mass logg spt band apmag vsini mdot vw tcoro B0 r0 tilt per

Column units used by pyGrater:

- dist in pc
- temp in K
- rad in solar radii
- mass in solar masses
- logg in cgs
- apmag in magnitudes (for the chosen band)
- vsini in km/s
- mdot in units of 1e-14 solar masses/year
- vw in m/s
- tcoro in K
- B0, r0, tilt, per are read as numeric values and can be set to nan if unused

### Example row

```text
MyStar 42.0 6500 1.3 1.2 4.1 F5 V 6.2 15.0 nan nan nan nan nan nan nan
```

Then load it in Python:

```python
from pyGrater import Star

star = Star(star_name="MyStar")
print(star.temp, star.distance)
```

If the name does not match exactly, pyGrater raises:

- ValueError: Unknown star: <name>



## Examples

Notebook examples are available in the examples directory, including:

- grain temperatures,
- flux profiles,
- SED and image generation.

---

## Logging

Every `print()` call made anywhere in pyGrater is automatically mirrored to a
timestamped log file whenever the package is imported.

### Default log location

Log files are written to a `logs/` folder **inside the current working
directory** at the moment `import pyGrater` is executed:

```
<cwd>/logs/pyGrater_YYYYMMDD_HHMMSS.log
```

So if you launch Python from `/home/user/my_project/`, the log appears at:

```
/home/user/my_project/logs/pyGrater_20260616_142301.log
```

### Changing the log directory

Pass a custom path to `redirect_print_to_log()` **before** (or right after)
importing pyGrater:

```python
import pyGrater
pyGrater.redirect_print_to_log("/path/to/my/logs")
```

If pyGrater has already been imported (e.g. in a Jupyter notebook that was not
restarted), call it again with the new path — the previous log file is closed
and a fresh one is opened at the new location.

### Disabling log file output

To suppress file logging for the current session, restore the original stdout:

```python
import sys
sys.stdout = sys.stdout._original   # unwrap the TeeStream
```


## Troubleshooting

- FileNotFoundError for data path:
  configure PYGRATER_DATA_PATH or run pygrater-setup --data-path.
- Unknown star name:
  check spelling and ensure the row exists in stars_main_properties.txt.
- Slow first run for a composition:
  grain efficiency files may need to be computed once before reuse.
