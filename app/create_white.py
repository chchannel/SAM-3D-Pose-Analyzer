import os
from PIL import Image

def create_white_test():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "test_white.jpg")
    img = Image.new("RGB", (512, 512), (255, 255, 255))
    img.save(path, "JPEG")
    print(f"✅ Created white test image: {path}")

if __name__ == "__main__":
    create_white_test()
