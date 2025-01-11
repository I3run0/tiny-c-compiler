# Tiny C Compiler

This repository contains the Tiny C Compiler, developed as part of the MC921 (Compilers) course at Unicamp. The compiler was implemented using Python and the SLY framework. You can run and debug Tiny C code using the `uc_code.py` script. Below, you'll find detailed instructions on how to set up and use the project.

---

## Requirements

- **Python**: Version 3.10 or newer.  
- **Pip Packages**:  
  - `sly`
  - `pytest`
  - `setuptools`
  - `graphviz`

---

## Installation

### Local Environment

We recommend using a virtual environment to isolate the project dependencies from your system packages. Follow these steps to install the project:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Docker

Alternatively, you can use Docker and Docker Compose (requires Docker Engine 19.03.0+) to set up the environment without modifying your local system.

---

## Usage

### Running the Compiler

You can execute the `uc_code.py` script directly with Python. To see the available options, run:

```sh
python3 uc/uc_code.py -h
```

If you have installed the project in your environment, you can use the `uc-code` command to run the code generator:

```sh
uc-code -h
```

To test the compiler with one of the sample inputs, use the files provided in the `tests/in-out/` directory. For example:

```sh
uc-code tests/in-out/t01.in
```

Additionally, the `uc_compiler.py` script provides a CLI interface through the `ucc` command. For more details:

```sh
ucc -h
```

### Using Docker

If you’re working in the Docker environment, you can run `uc_code.py` with the following commands:

```sh
docker-compose run --rm test uc/uc_code.py -h
```

Example of running a test file:

```sh
docker-compose run --rm test uc/uc_code.py tests/in-out/t01.in
```

---

## Testing with Pytest

You can automatically test the compiler using the `pytest` framework with the files in the `tests/in-out/` directory.

### Preparing for Testing

Make the source files accessible to the tests using one of these methods:

1. **Install the project in editable mode**:
   ```sh
   pip install -e .
   ```

2. **Add the repository to `PYTHONPATH`**:
   ```sh
   source setup.sh
   ```

### Running Tests Locally

Once the setup is complete, run all tests from the root of the repository:

```sh
pytest
```

### Running Tests with Docker

In the Docker environment, use this command to run all tests:

```sh
docker compose run --rm pytest
```

---

## Linting and Formatting (Optional)

For maintaining code quality, use the following tools:

- **Linting**: `flake8`
- **Formatting**: `black` and `isort`

### Installing Development Tools

Install the additional tools in the virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Linting

Run the following commands from the root of the repository to check for errors and coding style:

```sh
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-line-length=120 --statistics
```

The first command highlights errors that need to be fixed. The second provides warnings for improving code style.

### Formatting

To format the code, use the following commands:

```sh
isort .
black .
```