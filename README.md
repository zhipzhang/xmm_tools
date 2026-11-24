# xmm-tools

A Python helper package for XMM-Newton X-ray data analysis.

## Installation

Install in editable mode for development:

```bash
pip install -e ".[dev]"
```

Or with `uv`:

```bash
uv pip install -e ".[dev]"
```

## Project Structure

```text
xmm_tools/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── xmm_tools/
│       └── __init__.py
└── tests/
    └── test_package.py
```

## Running Tests

```bash
pytest
# or without pytest installed:
python3 -m unittest discover -s tests
```
