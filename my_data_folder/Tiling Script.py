"""
Satellite Image Tiling Script
=============================
Divides large satellite images into smaller tiles for training machine learning models.
Preserves geospatial metadata (CRS, transforms) in output tiles.

Functions:
    tile_satellite_image: Split large GeoTIFF images into manageable tile sizes
"""

import rasterio
from rasterio.windows import Window
import os

def tile_satellite_image(input_path, output_dir, tile_size=512):
    """Divide large satellite image into tiled GeoTIFF files.
    
    Splits large satellite imagery (e.g., THEOS-2) into smaller tiles for:
    - Model training (prevents memory issues)
    - Inference on high-resolution data
    - Parallel processing
    
    Args:
        input_path (str): Path to input satellite image (GeoTIFF format)
        output_dir (str): Directory to save output tiles
        tile_size (int, optional): Tile dimensions (square, default 512 pixels)
    
    Saves:
        Multiple GeoTIFF files named: tile_{x}_{y}.tif
        Each tile preserves original CRS, geospatial transform, and metadata
    
    Note:
        - Creates output directory if it doesn't exist
        - Tiles at image boundaries are smaller if not perfectly divisible
        - Each tile retains full geospatial metadata for accuracy
    
    Example:
        tile_satellite_image('THEOS2_POST_FLOOD.tif', 'train/images_post')
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with rasterio.open(input_path) as src:
        # Loop ตัดแบ่งภาพตามขนาดที่กำหนด
        for y in range(0, src.height, tile_size):
            for x in range(0, src.width, tile_size):
                # ตรวจสอบว่าขอบภาพต้องไม่เกินขนาดจริง
                window = Window(x, y, tile_size, tile_size)
                transform = src.window_transform(window)
                
                # อ่านข้อมูล
                tile = src.read(window=window)
                
                # บันทึกเป็นไฟล์ใหม่
                profile = src.profile
                profile.update({
                    'height': tile_size,
                    'width': tile_size,
                    'transform': transform
                })
                
                output_filename = f"{output_dir}/tile_{x}_{y}.tif"
                with rasterio.open(output_filename, 'w', **profile) as dst:
                    dst.write(tile)
    
    print(f"✅ หั่นภาพเสร็จสิ้น! บันทึกที่โฟลเดอร์: {output_dir}")

# ตัวอย่างการเรียกใช้:
# tile_satellite_image('THEOS2_POST_FLOOD.tif', 'train/images_post')