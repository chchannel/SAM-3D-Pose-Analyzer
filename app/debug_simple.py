import os
import sys

def check_file_exists(path):
    print(f"Checking: {path}")
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"✅ Found! Size: {size} bytes")
        # Read first few bytes to check if it's a valid JPEG
        with open(path, "rb") as f:
            header = f.read(10)
            print(f"Header: {header.hex()}")
    else:
        print("❌ NOT FOUND.")

if __name__ == "__main__":
    # Check the latest generated file in uploads
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    files = [f for f in os.listdir(target_dir) if f.endswith(".jpg")]
    if files:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(target_dir, x)), reverse=True)
        check_file_exists(os.path.join(target_dir, files[0]))
    else:
        print("No JPG files in uploads.")
