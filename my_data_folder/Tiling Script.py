import rasterio
from rasterio.windows import Window
import os

def tile_satellite_image(input_path, output_dir, tile_size=512):
    """
    หั่นภาพ THEOS-2 ขนาดใหญ่เป็น Tile เล็กๆ สำหรับเทรน AI
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