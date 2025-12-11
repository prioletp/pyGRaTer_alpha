## Installation

pip install pyGrater
(or install in editable mode: cd pyGrater folder | pip install -e .)


## Data Setup

Download the data folder from [this link](https://osf.io/mqkyf/overview) and configure the path:

import pyGrater
pyGrater.set_data_path('/path/to/downloaded/data')

Or set environment variable:

export PYGRATER_DATA_PATH=/path/to/downloaded/data

or run (only works if you pip installed): 

pygrater-setup --data-path /path/to/data