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


## How to Use a Star

### Option 1: load from the catalog

The simplest way is to pass the star's name — pyGrater looks it up in
`data/star_data/stars_main_properties.txt`:

```python
from pyGrater import Star

star = Star(star_name="bPic")
print(star.temp, star.distance)
```

If the name does not match exactly, pyGrater raises `ValueError: Unknown star: <name>`.

### Option 2: define a star inline (no catalog entry needed)

You can create a `Star` object entirely from keyword arguments, without
adding any row to the catalog.  The required kwargs are:

| kwarg  | unit          | description                        |
|--------|---------------|------------------------------------|
| `dist` | pc            | distance to the star               |
| `temp` | K             | effective temperature              |
| `rad`  | R☉            | stellar radius                     |
| `logg` | cgs           | surface gravity                    |
| `band` | —             | photometric band for normalization |
| `apmag`| mag           | apparent magnitude in that band    |
| `spt`  | —             | spectral type *(optional)*         |

```python
from pyGrater import Star

star = Star(
    dist  = 19.3,    # pc
    temp  = 8052,    # K
    rad   = 1.8,     # R_sun
    logg  = 4.1,     # cgs
    band  = "V",
    apmag = 3.86,    # mag
    spt   = "A6V",   # optional
)
print(star.temp, star.lum)
```

This is useful for quick tests, one-off targets, or stars not yet in the catalog.

### Adding a new star to the catalog (permanent)

To make a star permanently available by name, use the `add_star()` helper:

```python
from pyGrater.add_stars import add_star

add_star(
    star  = "MyStar",   # required — no spaces
    dist  = 42.0,       # required — pc
    temp  = 6500,       # required — K
    rad   = 1.3,        # required — R_sun
    logg  = 4.1,        # required — CGS
    band  = "V",        # required — photometric band
    apmag = 6.2,        # required — apparent magnitude
    mass  = 1.2,        # optional — M_sun
    spt   = "F5V",      # optional — spectral type
    vsini = 15.0,       # optional — km/s
    # all other fields default to nan
)
```

Or from the command line:

```bash
python -m pyGrater.add_stars \
  --star MyStar --dist 42.0 --temp 6500 --rad 1.3 --logg 4.1 \
  --band V --apmag 6.2 --spt F5V
```

The function raises `ValueError` if the star name already exists.
All unspecified optional columns (`mass`, `vsini`, `mdot`, `vw`, `tcoro`,
`B0`, `r0`, `tilt`, `per`) default to `nan`.

---

## How to Add a New Grain Material

> **Before registering a new material**, place its optical index file(s) in:
> ```
> data/optical_properties/
> ```
> The filenames you pass to `file_par`, `file_per1`, and `file_per2` are looked
> up relative to that folder.

Use the `add_material()` helper to append a new entry to
`data/material_list.txt`:

```python
from pyGrater.add_materials import add_material

# Minimal — single optical file for all three orientations
add_material(
    nickname = "my_dust",   # required — no spaces
    Tsub     = 1700,        # required — K
    density  = 3.5,         # required — g/cm³
    file_par = "my_dust.txt",  # required; per1 & per2 default to this
)

# With separate orientation files and metadata
add_material(
    nickname   = "my_dust",
    Tsub       = 1700,
    density    = 3.5,
    file_par   = "my_dust_par.txt",
    file_per1  = "my_dust_per1.txt",
    file_per2  = "my_dust_per2.txt",
    wav_min    = 0.2,
    wav_max    = 500.0,
    full_name  = "My custom silicate",
    formula    = "MgSiO3",
    reference  = "Author et al. 2025",
)
```

Or from the command line:

```bash
python -m pyGrater.add_materials \
  --nickname my_dust --Tsub 1700 --density 3.5 \
  --file_par my_dust.txt \
  --full_name "My custom silicate" --formula MgSiO3
```

**Key behaviours:**
- Required: `nickname`, `Tsub`, `density`, `file_par`
- If `file_per1` / `file_per2` are omitted, all three orientations use `file_par`
- Orientation weights default to `0.333333` each
- Raises `ValueError` if the nickname already exists




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
