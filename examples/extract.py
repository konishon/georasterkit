import logging
from pathlib import Path
from georasterkit.tiff_extractor import TiffExtractor

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    base_dir = Path(__file__).parent.parent
    input_tiff = base_dir / "sample_data" / "1.tif"
    output_dir = base_dir / "tiles"

    extractor = TiffExtractor(
        tiff_path=input_tiff,
        tile_size=(2048, 2048),          # width, height in pixels
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

if __name__ == "__main__":
    main()
