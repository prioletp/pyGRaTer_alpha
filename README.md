## Installation

Install **pyGrater** using pip:

```bash
pip install pyGrater
```

Or install in editable (developer) mode:

```bash
cd /path/to/pyGrater
pip install -e .
```

---

## Data Setup

Download the required **data folder** from:  
https://osf.io/mqkyf/overview

After downloading, set the data path using one of the methods below.

### 1. Set the path inside Python

```python
import pyGrater
pyGrater.set_data_path("/path/to/downloaded/data")
```

### 2. Set an environment variable

```bash
export PYGRATER_DATA_PATH=/path/to/downloaded/data
```

### 3. Use the command-line setup helper (only works if installed via pip)

```bash
pygrater-setup --data-path /path/to/data
```
