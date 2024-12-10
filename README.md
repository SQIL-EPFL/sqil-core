# For users
## Installation
```bash
$ pip install sqil_core
```

## Usage
You can find all the functions available and examples in the documentation.
```python
import sqil_core as sqil

path = 'path to your data folder'

# Extract data
mag, phase, freq = sqil.extract_h5_data(path, ['mag_dB', 'phase', 'ro_freq'])
```

# For developers
## Development
Install poetry if you haven't already (`pip install poetry`) and then run the following
```bash
$ poetry install
$ poetry run pre-commit install
```

#### Test your changes
```bash
$ pip install -e . --user
```
If you're using a jupyter notebook remember to restart the kernel

## Build
```bash
$ poetry run build
```

## Docs
Serve docs
```bash
$ poetry run docs_serve
```

Build docs
```bash
$ poetry run docs_build
```
