# GeoRasterKit

A suite of utilities for slicing, masking, and merging overlapping GeoTIFF rasters.

---

## Features

- **Tile**: Split large GeoTIFFs into smaller, optionally overlapping tiles.
- **Mask**: Join binary masks with GeoTIFF data.
- **Merge**: Combine overlapping tiles or rasters back into a single file.

## Project Layout

```bash
georasterkit/            # Core package
├── __init__.py
└── tiff_extractor.py

examples/                # Example scripts
└── example.py

tests/                   # Pytest suite
└── test_tiff_extractor.py

sample_data/             # Sample GeoTIFFs for testing/demo


README.md
LICENSE
pyproject.toml
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/naxa-developers/georasterkit.git
cd georasterkit
```

### 2. Using Poetry

```bash
poetry install
poetry shell
```



## Quick Start

```python
from georasterkit.tiff_extractor import TiffExtractor

extractor = TiffExtractor(
    tiff_path="sample_data/1.tif",
    tile_size=256,
    overlap_percentage=10.0,
    output_folder="tiles",
    remove_folder_if_exist=True,
    skip_if_output_exists=False,
    debug=True,
)

success = extractor.extract()
if success:
    print("Tiles generated in 'tiles/' folder.")
```


