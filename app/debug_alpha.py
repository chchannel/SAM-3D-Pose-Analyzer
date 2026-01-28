import os
import sys
from PIL import Image
import numpy as np

def debug_image(image_path):
    print(f"--- Debugging: {image_path} ---")
    if not os.path.exists(image_path):
        print("❌ File NOT found.")
        return

    try:
        img = Image.open(image_path)
        print(f"Mode: {img.mode}")
        print(f"Format: {img.format}")
        print(f"Size: {img.size}")
        
        # Check some sample pixels (corners tend to be background)
        data = np.array(img)
        corners = [data[0,0], data[0,-1], data[-1,0], data[-1,-1]]
        print(f"Corner pixels (RGB): {corners}")
        
        # Explicit test of the conversion logic
        print("\n[Simulation of Conversion Logic]")
        img_rgba = img.convert("RGBA")
        canvas = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(canvas, img_rgba)
        final_rgb = composite.convert("RGB")
        
        final_data = np.array(final_rgb)
        final_corners = [final_data[0,0], final_data[0,-1], final_data[-1,0], final_data[-1,-1]]
        print(f"Post-Conversion Corner pixels (RGB): {final_corners}")
        
        if np.all(final_corners[0] == [255, 255, 255]):
            print("✅ Result: Background is WHITE (255, 255, 255)")
        else:
            print(f"❌ Result: Background is NOT white. It is {final_corners[0]}")

    except Exception as e:
        print(f"❌ Error during debug: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_image(sys.argv[1])
    else:
        # Check the latest one from the logs if available
        test_path = "/home/bboy/MPPA/app/uploads/input_rec_1769507006228_mppa_cv_.jpg"
        debug_image(test_path)
