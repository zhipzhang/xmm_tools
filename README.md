# xmm-tools

A modular Python toolkit for XMM-Newton X-ray extended-source analysis,
bridging SAS/ESAS data reduction, interactive DS9 region & QDP inspection, and
transparent OOP Sherpa joint spectral fitting.

## Installation

Core ESAS & Sherpa wrappers (also works inside headless `xmmsas` Docker containers):

```bash
pip install -e .
```

With interactive GUI viewers (`xmm-qdp` and `xmm-img`):

```bash
pip install -e ".[gui]"
```

With development or documentation dependencies:

```bash
pip install -e ".[dev,docs]"
```

## Quick Start

```bash
# 1. One-time ODF initialization (generates esas.toml + standardized *_evt.fits)
xmm-esas init --odf-dir /path/to/ODF --workdir ./esas_work --ccfpath /ccf/valid
cd esas_work

# 2. Run all config-driven stages automatically (flare -> cheese -> quick-image -> spectrum)
#    (spectrum is skipped cleanly if [[spectrum.regions]] is not configured yet)
xmm-esas run --config esas.toml

# 3. Or run specific stages / a stage range:
xmm-esas run --config esas.toml --stages flare,cheese
xmm-esas run --config esas.toml --from cheese --to spectrum
xmm-esas run --config esas.toml --skip quick-image

# 4. Or run any individual stage directly:
xmm-esas flare --config esas.toml
xmm-esas cheese --config esas.toml
xmm-esas quick-image --config esas.toml
xmm-esas spectrum --config esas.toml
```

Interactive viewers:

```bash
xmm-qdp soft_proton_flare/mos1_evt-hist.qdp
xmm-img quick_image/soft_500_2000_flux.fits
```

## Documentation

Build the Sphinx HTML documentation locally (includes a full annotated
`esas.toml` reference, pipeline guide, GUI workflows, and Sherpa joint-fitting
examples):

```bash
sphinx-build -b html docs docs/_build/html
```

## Running Tests

```bash
pytest
# or without pytest installed:
python3 -m unittest discover -s tests
```
