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
# 1. One-time ODF initialization (generates ana.toml, init_manifest.json, + standardized *_evt.fits)
#    Automatically selects unique scheduled exposures (S\d{3}); pass --select INSTRUMENT=FILENAME to override
xmm-ana init --odf-dir /path/to/ODF --workdir ./esas_work --ccfpath /ccf/valid
cd esas_work

# 2. Run all config-driven stages automatically (flare -> cheese -> qpb-image -> spectrum)
#    (spectrum is skipped cleanly if [[esas.spectrum.regions]] is not configured yet;
#     unchanged steps are reused automatically via schema v2 manifest.json SHA-256 ledgers)
xmm-ana run --config ana.toml

# 3. Or run specific stages / a stage range:
xmm-ana run --config ana.toml --stages flare,cheese
xmm-ana run --config ana.toml --from cheese --to spectrum
xmm-ana run --config ana.toml --skip qpb-image

# 4. Or run any individual stage directly:
xmm-ana flare --config ana.toml
xmm-ana cheese --config ana.toml
xmm-ana qpb-image --config ana.toml
xmm-ana spectrum --config ana.toml
```

Interactive viewers:

```bash
xmm-qdp soft_proton_flare/mos1_evt-hist.qdp
xmm-img qpb_image/soft_500_2000_flux.fits
```

## Documentation

Build the Sphinx HTML documentation locally (includes a full annotated
`ana.toml` reference, pipeline guide, `QPBProducts` usage, GUI workflows, and
Sherpa joint-fitting examples):

```bash
sphinx-build -b html docs docs/_build/html
```

## Running Tests

```bash
pytest
# or without pytest installed:
python3 -m unittest discover -s tests
```
