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

1. **Install poetry if you haven't already**
```bash
$ pip install poetry
```

2. **Install the required packages using poetry**
```bash
$ poetry install
```

3. **Install the pre-commit hooks**
If you are on windows you need to install git ([https://git-scm.com/downloads](here)) and add it to your windows PATH.
After the installation open a new terminal.
```bash
$ poetry run pre-commit install
```
This will check if your python files are formatted correctly when you try to commit.
If that's not the case the commit will be canceled and the files will be automatically formatted.
Then you'll have to add and commit again the new files.

4. **Start the virtual environment**
```bash
$ poetry shell
```
To exit the virtual environment just use `exit`

#### Test your changes

```bash
$ pip install -e . --user
```

**Anaconda**
If you want to install in a specific anaconda environment

- from your poetry shell build the package

```bash
$ poetry run build
```

- open an anaconda shell
- activate the desired environemnt
- pip install the wheel file (.whl) in the dist folder of the sqil-core project

```bash
$ pip install PATH_TO_SQIL_CORE_FOLDER/dist/SQIL_CORE-VERSION.whl
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
