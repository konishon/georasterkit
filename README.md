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
import logging
from pathlib import Path
from georasterkit.tiff_extractor import TiffExtractor

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

base_dir = Path(__file__).parent.parent
input_tiff = base_dir / "sample_data" / "1.tif"
output_dir = base_dir / "tiles"

extractor = TiffExtractor(
    tiff_path=input_tiff,
    tile_size=(256, 256),          # width, height in pixels
    overlap=(0.1, 0.1),            # 10% overlap x and y
    output_folder=output_dir,
    force=True,
    workers=4,
    debug=True,
)

success = extractor.extract()
if success:
    logger.info(
        "Tiles written to '%s'. Grid preview (grid_preview.png) also generated.",
        output_dir
    )
else:
    logger.error("Tile extraction failed.")


```


