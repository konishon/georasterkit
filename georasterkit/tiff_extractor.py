import logging
import shutil
import time
from pathlib import Path
from typing import Union, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rasterio.windows import Window
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TiffExtractor:
    """
    Extract tiles from a GeoTIFF with optional distinct horizontal and vertical overlaps, plot grid preview, and multithreading.

    Attributes:
        tile_size (Tuple[int,int]): Tile width and height in pixels.
        overlap (Tuple[float,float]): Overlap fractions in x and y (0–1).
    """

    def __init__(
        self,
        tiff_path: Union[str, Path],
        tile_size: Union[int, Tuple[int, int]],
        overlap: Union[float, Tuple[float, float]] = 0.0,
        output_folder: Union[str, Path] = "./tiles",
        force: bool = False,
        workers: int = 1,
        debug: bool = False,
    ) -> None:
        self.tiff_path = Path(tiff_path)
        if isinstance(tile_size, int):
            self.tile_width = self.tile_height = tile_size
        else:
            self.tile_width, self.tile_height = tile_size
        if isinstance(overlap, tuple):
            self.overlap_x, self.overlap_y = overlap
        else:
            self.overlap_x = self.overlap_y = overlap
        self.output_folder = Path(output_folder)
        self.force = force
        self.workers = workers

        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=level)

        if self.tile_width <= 0 or self.tile_height <= 0:
            raise ValueError("tile dimensions must be positive")
        for ov in (self.overlap_x, self.overlap_y):
            if not (0 <= ov < 1):
                logger.warning("overlap fractions should be in [0,1), got %s", ov)
        if workers < 1:
            raise ValueError("workers must be >= 1")

    def extract(self) -> bool:
        start = time.perf_counter()
        if self.output_folder.exists():
            if self.force:
                shutil.rmtree(self.output_folder)
            else:
                logger.error("Output '%s' exists. Use force=True to overwrite.", self.output_folder)
                return False
        self.output_folder.mkdir(parents=True, exist_ok=True)

        try:
            with rasterio.open(self.tiff_path) as src:
                if not self._validate_src(src):
                    return True
                w, h, count = src.width, src.height, src.count
                step_x = max(1, int(self.tile_width * (1 - self.overlap_x)))
                step_y = max(1, int(self.tile_height * (1 - self.overlap_y)))
                cols = list(range(0, w, step_x))
                rows = list(range(0, h, step_y))
                total = len(rows) * len(cols)
                logger.info(
                    "Image size: %dx%d pixels", w, h
                )
                logger.info(
                    "Tile size: %dx%d, overlaps: %.2f%% x, %.2f%% y", 
                    self.tile_width, self.tile_height, 
                    self.overlap_x*100, self.overlap_y*100
                )
                logger.info(
                    "Grid: %d cols x %d rows = %d tiles (step_x=%d, step_y=%d)",
                    len(cols), len(rows), total, step_x, step_y
                )
                self._plot_preview(w, h, rows, cols)
                tasks = [(r, c) for r in rows for c in cols]

            if self.workers > 1:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    futures = [executor.submit(self._process_tile, r, c, src_count=count) for r, c in tasks]
                    for _ in tqdm(as_completed(futures), total=total, desc="Tiles"):
                        pass
            else:
                for r, c in tqdm(tasks, total=total, desc="Tiles"):
                    self._process_tile(r, c, src_count=count)

            elapsed = time.perf_counter() - start
            logger.info("Extraction complete in %.2f sec", elapsed)
            return True

        except Exception:
            logger.exception("Extraction failed")
            return False

    def _validate_src(self, src: rasterio.io.DatasetReader) -> bool:
        w, h, c = src.width, src.height, src.count
        if w == 0 or h == 0 or c == 0:
            logger.warning("Empty TIFF (%dx%d, bands=%d)", w, h, c)
            return False
        return True

    def _plot_preview(self, width: int, height: int, rows: List[int], cols: List[int]) -> None:
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        for r in rows:
            for c in cols:
                rect = Rectangle((c, r), self.tile_width, self.tile_height, linewidth=1, 
                                 edgecolor='black', facecolor='none', alpha=0.5)
                ax.add_patch(rect)
                ox = int(self.tile_width * self.overlap_x)
                oy = int(self.tile_height * self.overlap_y)
                if ox:
                    rect_h = Rectangle((c + self.tile_width - ox, r), ox, self.tile_height,
                                       facecolor='red', alpha=0.3)
                    ax.add_patch(rect_h)
                if oy:
                    rect_v = Rectangle((c, r + self.tile_height - oy), self.tile_width, oy,
                                       facecolor='blue', alpha=0.3)
                    ax.add_patch(rect_v)
        ax.axis('off')
        preview_path = self.output_folder / 'grid_preview.png'
        fig.savefig(preview_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        logger.info("Grid preview saved to %s", preview_path)

    def _process_tile(self, row_off: int, col_off: int, src_count: int) -> None:
        try:
            with rasterio.open(self.tiff_path) as src:
                tw = min(self.tile_width, src.width - col_off)
                th = min(self.tile_height, src.height - row_off)
                if tw <= 0 or th <= 0:
                    return
                window = Window(col_off, row_off, tw, th)
                bands = list(range(1, min(3, src_count) + 1))
                data = src.read(bands, window=window)
                meta = src.meta.copy()
                meta.update({
                    "driver": "GTiff",
                    "height": th,
                    "width": tw,
                    "transform": src.window_transform(window),
                    "count": len(bands),
                    "dtype": data.dtype,
                })
                if src.nodata is not None:
                    meta["nodata"] = src.nodata
                out_path = self.output_folder / f"tile_{row_off}_{col_off}.tif"
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(data)
        except Exception:
            logger.warning("Tile (%d,%d) failed", row_off, col_off, exc_info=True)
