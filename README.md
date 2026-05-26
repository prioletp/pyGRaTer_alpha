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

Download the required **data folder** from:  
https://osf.io/mqkyf/overview

After downloading, set the data path using one of the methods below.

### Set the path inside Python

```python
import pyGrater
pyGrater.set_data_path("/path/to/downloaded/data")
```

### Set an environment variable

```bash
export PYGRATER_DATA_PATH=/path/to/downloaded/data
```

### Use the command-line setup helper (only works if installed via pip)

```bash
pygrater-setup --data-path /path/to/data
```
