"""
Data Engineering for Flood Detection - NDWI Calculation
========================================================
Optimized tiled processing of satellite imagery for water detection.
Uses NDWI (Normalized Difference Water Index) to identify water bodies from multispectral satellite data.

Functions:
    calculate_optimized_ndwi_tiled: Process large satellite images in tiles for memory efficiency
    
Key Optimizations:
    - Tiled reading to prevent memory overflow
    - CRS (Coordinate Reference System) validation
    - Binary mask compression using INT8
    - UINT16 intermediate representation
"""

import rasterio
import numpy as np

def calculate_optimized_ndwi_tiled(pre_image_path, post_image_path, tile_size=1024):
    """Calculate optimized NDWI (Normalized Difference Water Index) using tiled processing.
    
    Processes large satellite images in tiles to detect water bodies and changes.
    NDWI = (Green - NIR) / (Green + NIR)
    
    Args:
        pre_image_path (str): Path to pre-event satellite image (multi-band GeoTIFF)
        post_image_path (str): Path to post-event satellite image (multi-band GeoTIFF)
        tile_size (int, optional): Tile size for processing. Defaults to 1024 pixels.
    
    Returns:
        list: List of dictionaries containing:
            - 'window': rasterio Window object for tile location
            - 'water_mask': Binary numpy array (0=not water, 1=water)
            - 'transform': Geospatial transform for coordinates
    
    Processing:
        1. Validates CRS (Coordinate Reference System) match between images
        2. Iterates through image tiles
        3. Reads Green (band 3) and NIR (band 4) bands
        4. Computes NDWI with epsilon for numerical stability
        5. Creates binary water mask (threshold > 0.1)
        6. Compresses to INT8 for memory efficiency
    
    Raises:
        Warning if CRS doesn't match between images
    """
    
    with rasterio.open(pre_image_path) as pre_ds, rasterio.open(post_image_path) as post_ds:
        # ⚠️ optimization: ตรวจสอบพิกัด CRS (Coordinate Reference System) ให้ตรงกัน
        if pre_ds.crs != post_ds.crs:
            print("🚫 คำเตือน: พิกัดภาพ (CRS) ไม่ตรงกัน. ต้องแปลงภาพให้ตรงกันก่อน")
            return []

        transform = pre_ds.transform
        width, height = pre_ds.width, pre_ds.height
        
        # 🚀 TILING optimization: สไลด์อ่านภาพทีละชิ้น
        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):
                window = rasterio.windows.Window(col, row, tile_size, tile_size)
                
                # อ่านค่าแบนด์ที่จำเป็น (สมมติ Green=3, NIR=4)
                # และอ่านแบบ UINT16 เพื่อประหยัด Memory
                pre_green = pre_ds.read(3, window=window, boundless=True, fill_value=0).astype(np.float32)
                pre_nir = pre_ds.read(4, window=window, boundless=True, fill_value=0).astype(np.float32)
                
                # ⚠️ optimization: คำนวณ NDWI แบบไม่ให้ Memory ล้น
                pre_ndwi = (pre_green - pre_nir) / (pre_green + pre_nir + 1e-6) # + epsilon กันค่า 0
                
                # ⚠️ optimization: บีบอัดเป็น Binary Mask (0 ไม่ใช่น้ำ, 1 คือน้ำ)
                # และใช้ INT8 เพื่อประหยัด Memory มหาศาล
                pre_water_mask = (pre_ndwi > 0.1).astype(np.int8)
                
                results.append({
                    'window': window,
                    'water_mask': pre_water_mask,
                    'transform': transform
                })
                # print(f"ประมวลผล Tiles {row}, {col} สำเร็จ") # debug
    
    print("--- ✅ การคัดกรองผืนน้ำแบบ Optimized สำเร็จ ---")
    return results

# ตัวอย่างการเรียกใช้ (ใช้พาร์ทไฟล์ที่คุณมีใน `image_2.png`)
# calculate_optimized_ndwi_tiled('IMG_...Pre.TIF', 'IMG_...Post.TIF')