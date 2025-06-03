import logging
import shutil
import time
from pathlib import Path
from typing import Union, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import rasterio
import matplotlib
matplotlib.use('Agg') # Ensure non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rasterio.windows import Window
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class TiffExtractor:
    """
    Extract tiles from a GeoTIFF with optional distinct horizontal and vertical overlaps, plot grid preview, and multithreading.

    Attributes:
        tile_size (Tuple[int,int]): Tile width and height in pixels.
        overlap (Tuple[float,float]): Overlap fractions in x and y (0–1).
    """
    # Constants for preview plotting
    _PREVIEW_MAX_DIM_PX = 2048  # Maximum dimension (width or height) for the preview image in pixels
    _PREVIEW_DPI = 100         # DPI for the saved preview image

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

        current_level = logging.DEBUG if debug else logging.INFO
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=current_level,
                format="%(asctime)s [%(levelname)s] %(message)s (%(name)s)",
            )
        else:
            logger.setLevel(current_level)


        if self.tile_width <= 0 or self.tile_height <= 0:
            raise ValueError("tile dimensions must be positive")
        for ov_val in (self.overlap_x, self.overlap_y):
            if not (0 <= ov_val < 1): # Changed 'ov' to 'ov_val' to avoid conflict if used in a loop later
                logger.warning("overlap fractions should be in [0,1), got %s", ov_val)
        if workers < 1:
            raise ValueError("workers must be >= 1")

    def extract(self) -> bool:
        start_time = time.perf_counter() # Renamed 'start' to 'start_time'
        if self.output_folder.exists():
            if self.force:
                logger.info("Force enabled: Removing existing output folder '%s'", self.output_folder)
                shutil.rmtree(self.output_folder)
            else:
                logger.error("Output folder '%s' already exists. Use force=True to overwrite.", self.output_folder)
                return False
        try:
            self.output_folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error("Failed to create output folder '%s': %s", self.output_folder, e)
            return False


        try:
            with rasterio.open(self.tiff_path) as src:
                if not self._validate_src(src):
                    # If validation fails (e.g. empty TIFF), extraction should be considered failed.
                    return False 
                
                img_width, img_height, band_count = src.width, src.height, src.count # Renamed for clarity
                
                step_x = max(1, int(self.tile_width * (1 - self.overlap_x)))
                step_y = max(1, int(self.tile_height * (1 - self.overlap_y)))
                
                cols = list(range(0, img_width, step_x))
                rows = list(range(0, img_height, step_y))
                
                if not cols or not rows:
                    logger.warning("Calculated grid has zero columns or rows. Check tile size and image dimensions.")
                    return False 

                total_tiles = len(rows) * len(cols) # Renamed 'total'
                
                logger.info(
                    "Image size: %dx%d pixels, %d bands", img_width, img_height, band_count
                )
                logger.info(
                    "Tile size: %dx%d, overlaps: %.2f%% x, %.2f%% y",
                    self.tile_width, self.tile_height,
                    self.overlap_x*100, self.overlap_y*100
                )
                logger.info(
                    "Grid: %d cols x %d rows = %d tiles (step_x=%d, step_y=%d)",
                    len(cols), len(rows), total_tiles, step_x, step_y
                )
                
                self._plot_preview(img_width, img_height, rows, cols)
                
                tasks = [(r, c) for r in rows for c in cols]
                logger.info("Starting tile extraction for %d tasks...", len(tasks))
            if self.workers > 1:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    futures = [executor.submit(self._process_tile, r, c, src_band_count=band_count) for r, c in tasks]
                    for _ in tqdm(as_completed(futures), total=total_tiles, desc="Processing Tiles"):
                        pass # Results can be checked here if _process_tile returned success/failure
            else:
                for r, c in tqdm(tasks, total=total_tiles, desc="Processing Tiles (single-threaded)"):
                    self._process_tile(r, c, src_band_count=band_count)

            elapsed_time = time.perf_counter() - start_time
            logger.info("Extraction completed successfully in %.2f seconds.", elapsed_time)
            return True

        except FileNotFoundError:
            logger.error("Source TIFF file not found at '%s'", self.tiff_path)
            return False
        except rasterio.errors.RasterioIOError as e:
            logger.error("Rasterio I/O error processing '%s': %s", self.tiff_path, e)
            return False
        except Exception:
            logger.exception("An unexpected error occurred during tile extraction from '%s'", self.tiff_path)
            return False

    def _validate_src(self, src: rasterio.io.DatasetReader) -> bool:
        w, h, c = src.width, src.height, src.count
        if w == 0 or h == 0: # Allow c == 0 for metadata-only files if rasterio handles them? Typically c >= 1
            logger.warning("TIFF is empty or has zero dimensions (%dx%d pixels, %d bands). Cannot process.", w, h, c)
            return False
        if c == 0:
             logger.warning("TIFF has zero bands (%dx%d pixels, %d bands). Cannot process.", w, h, c)
             return False
        return True

    def _plot_preview(self, img_width: int, img_height: int, rows: List[int], cols: List[int]) -> None:
        if img_width == 0 or img_height == 0:
            logger.warning("Cannot generate grid preview for an image with zero width or height.")
            return

        img_aspect_ratio = float(img_width) / img_height

        if img_width > img_height:
            preview_width_px = min(img_width, self._PREVIEW_MAX_DIM_PX)
            preview_height_px = preview_width_px / img_aspect_ratio
        else:
            preview_height_px = min(img_height, self._PREVIEW_MAX_DIM_PX)
            preview_width_px = preview_height_px * img_aspect_ratio
        
        # Ensure dimensions are at least 1px for fig calculation and positive
        preview_width_px = max(1, int(round(preview_width_px)))
        preview_height_px = max(1, int(round(preview_height_px)))

        fig_width_inches = preview_width_px / self._PREVIEW_DPI
        fig_height_inches = preview_height_px / self._PREVIEW_DPI

        fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches), dpi=self._PREVIEW_DPI)
        
        ax.set_xlim(0, img_width)
        ax.set_ylim(0, img_height) # Origin will be bottom-left by default for plot
        ax.invert_yaxis()         # Standard for image display (origin top-left)
        ax.set_aspect('equal')    # Ensures pixels are square if data units are same scale

        for r_offset in rows: # Renamed 'r' to 'r_offset'
            for c_offset in cols: # Renamed 'c' to 'c_offset'
                rect = Rectangle(
                    (c_offset, r_offset), self.tile_width, self.tile_height,
                    linewidth=0.5, # Reduced linewidth for potentially dense grids
                    edgecolor='black', facecolor='none', alpha=0.5
                )
                ax.add_patch(rect)
                
                overlap_px_x = int(self.tile_width * self.overlap_x) # Renamed 'ox'
                overlap_px_y = int(self.tile_height * self.overlap_y) # Renamed 'oy'
                
                if overlap_px_x > 0:
                    rect_h_overlap = Rectangle(
                        (c_offset + self.tile_width - overlap_px_x, r_offset),
                        overlap_px_x, self.tile_height,
                        facecolor='red', alpha=0.2 # Reduced alpha
                    )
                    ax.add_patch(rect_h_overlap)
                if overlap_px_y > 0:
                    rect_v_overlap = Rectangle(
                        (c_offset, r_offset + self.tile_height - overlap_px_y),
                        self.tile_width, overlap_px_y,
                        facecolor='blue', alpha=0.2 # Reduced alpha
                    )
                    ax.add_patch(rect_v_overlap)
        
        ax.axis('off')
        preview_path = self.output_folder / 'grid_preview.png'
        try:
            fig.savefig(preview_path, bbox_inches='tight', pad_inches=0)
            logger.info("Grid preview saved to %s (%dx%d pixels)", preview_path, preview_width_px, preview_height_px)
        except Exception as e:
            logger.error("Failed to save grid preview image: %s", e)
        finally:
            plt.close(fig) # Ensure figure is closed to free memory

    def _process_tile(self, row_offset: int, col_offset: int, src_band_count: int) -> None: # Renamed params
        try:
            # Each thread/task opens the source file independently.
            # This is thread-safe but less performant than opening once and passing the dataset reader object.
            # However, for fixing memory primarily, this is acceptable.
            with rasterio.open(self.tiff_path) as src:
                # Calculate actual tile width and height, clipping at image boundaries
                actual_tile_width = min(self.tile_width, src.width - col_offset)
                actual_tile_height = min(self.tile_height, src.height - row_offset)

                if actual_tile_width <= 0 or actual_tile_height <= 0:
                    # This can happen if tile_offset is at or beyond the image edge,
                    # especially if step calculation leads to offsets starting outside.
                    # logger.debug("Skipping zero-dimension tile at (%d,%d)", row_offset, col_offset)
                    return

                window = Window(col_offset, row_offset, actual_tile_width, actual_tile_height)
                
                # Read up to 3 bands, or fewer if src_band_count is less.
                # Band indices in rasterio are 1-based.
                num_bands_to_read = min(3, src_band_count)
                bands_to_read = list(range(1, num_bands_to_read + 1))
                
                data = src.read(bands_to_read, window=window)
                
                # If, after reading, data is empty (e.g., all nodata or unexpected issue), skip.
                if data.size == 0:
                    logger.warning("Tile data read for window %s at (%d,%d) is empty. Skipping.", window, row_offset, col_offset)
                    return

                tile_transform = src.window_transform(window)
                
                tile_meta = src.meta.copy()
                tile_meta.update({
                    "driver": "GTiff",
                    "height": actual_tile_height,
                    "width": actual_tile_width,
                    "transform": tile_transform,
                    "count": len(bands_to_read), # Number of bands actually read and to be written
                    "dtype": data.dtype, # Use dtype of actual read data
                })
                
                # Preserve nodata value if it exists in source
                if src.nodata is not None:
                    tile_meta["nodata"] = src.nodata
                else: # If src.nodata is None, remove from meta if it was copied
                    tile_meta.pop("nodata", None)


                out_path = self.output_folder / f"tile_{row_offset}_{col_offset}.tif"
                with rasterio.open(out_path, "w", **tile_meta) as dst:
                    dst.write(data)
                    
        except Exception:
            # Log with exception info for debugging, but don't let one failed tile stop others.
            logger.warning(
                "Failed to process or write tile at offset (%d,%d) for %s",
                row_offset, col_offset, self.tiff_path.name,
                exc_info=True # Adds stack trace to log for this warning
            )