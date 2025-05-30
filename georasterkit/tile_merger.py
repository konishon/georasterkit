import logging
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class TileMerger:
    def __init__(
        self,
        tiles_folder: Path,
        tile_width: int,
        tile_height: int,
        overlap_x: float = 0.0,
        overlap_y: float = 0.0,
        weight_type: str = "hanning",
    ):
        self.tiles_folder = tiles_folder
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        self.weight_type = weight_type

    def create_weight_map(self, window_size: int) -> np.ndarray:
        """Create 1D weight vector according to the weight_type."""
        if self.weight_type == "hanning":
            return np.hanning(window_size)
        elif self.weight_type == "hamming":
            return np.hamming(window_size)
        elif self.weight_type == "gaussian":
            sigma = window_size / 6
            x = np.linspace(-window_size / 2, window_size / 2, window_size)
            gauss = np.exp(-0.5 * (x / sigma) ** 2)
            return gauss / gauss.max()
        elif self.weight_type == "uniform":
            return np.ones(window_size)
        else:
            raise ValueError(f"Unknown weight_type: {self.weight_type}")

    def merge(self, output_path: Path) -> bool:
        tile_files = list(self.tiles_folder.glob("tile_*.tif"))
        if not tile_files:
            logger.error("No tiles found in %s", self.tiles_folder)
            return False

        tile_offsets = []
        for f in tile_files:
            parts = f.stem.split("_")
            if len(parts) != 3:
                continue
            try:
                row_off = int(parts[1])
                col_off = int(parts[2])
                tile_offsets.append((row_off, col_off))
            except ValueError:
                continue

        max_row = max(r for r, _ in tile_offsets)
        max_col = max(c for _, c in tile_offsets)
        output_width = max_col + self.tile_width
        output_height = max_row + self.tile_height

        logger.info("Output raster size: %d x %d", output_width, output_height)
        logger.info("Using weight map type: %s", self.weight_type)

        with rasterio.open(tile_files[0]) as sample_tile:
            meta = sample_tile.meta.copy()
            meta.update({
                "width": output_width,
                "height": output_height,
                "count": sample_tile.count,
                "transform": sample_tile.transform,
            })

            origin_tile = next((f for f in tile_files if f.stem.endswith("_0_0")), None)
            if origin_tile:
                with rasterio.open(origin_tile) as origin_src:
                    base_transform = origin_src.transform
            else:
                base_transform = sample_tile.transform

            meta["transform"] = base_transform

            merged_data = np.zeros((meta["count"], output_height, output_width), dtype=meta["dtype"])
            weight_sum = np.zeros((output_height, output_width), dtype=np.float32)

        for tile_path in tqdm(tile_files, desc="Merging tiles"):
            parts = tile_path.stem.split("_")
            row_off = int(parts[1])
            col_off = int(parts[2])

            with rasterio.open(tile_path) as tile:
                data = tile.read()
                h, w = data.shape[1], data.shape[2]

                wx = self.create_weight_map(w)
                wy = self.create_weight_map(h)
                weight_patch = wy[:, None] * wx[None, :]

                for band_idx in range(meta["count"]):
                    merged_data[band_idx,
                                row_off:row_off + h,
                                col_off:col_off + w] += data[band_idx] * weight_patch

                weight_sum[row_off:row_off + h, col_off:col_off + w] += weight_patch

        normalized_data = merged_data / np.maximum(weight_sum, 1e-6)

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(normalized_data.astype(meta["dtype"]))

        logger.info("Merged GeoTIFF saved to %s", output_path)
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge GeoTIFF tiles with overlap blending.")
    parser.add_argument(
        "--tiles_folder",
        type=str,
        required=True,
        help="Folder path containing tiles named tile_{row}_{col}.tif",
    )
    parser.add_argument(
        "--tile_width",
        type=int,
        required=True,
        help="Tile width in pixels",
    )
    parser.add_argument(
        "--tile_height",
        type=int,
        required=True,
        help="Tile height in pixels",
    )
    parser.add_argument(
        "--overlap_x",
        type=float,
        default=0.0,
        help="Horizontal overlap fraction (0 to <1)",
    )
    parser.add_argument(
        "--overlap_y",
        type=float,
        default=0.0,
        help="Vertical overlap fraction (0 to <1)",
    )
    parser.add_argument(
        "--weight_type",
        type=str,
        choices=["hanning", "hamming", "gaussian", "uniform"],
        default="hanning",
        help="Type of weight map to use for blending tiles",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output merged GeoTIFF filepath",
    )
    args = parser.parse_args()

    merger = TileMerger(
        tiles_folder=Path(args.tiles_folder),
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        overlap_x=args.overlap_x,
        overlap_y=args.overlap_y,
        weight_type=args.weight_type,
    )

    merger.merge(Path(args.output_file))
