
import os
import sys
import subprocess

# Determine paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
LIB_DIR = os.path.join(BASE_DIR, "lib")
BLENDER_SCRIPT = os.path.join(LIB_DIR, "blender_bvh_crysta.py")
JSON_PATH = os.path.join(PARENT_DIR, "output_3_mhr70.json") # Look in parent
OUTPUT_BVH = os.path.join(PARENT_DIR, "crysta_motion.bvh")   # Output to parent for visibility

print(f">>> CRYSTAL BVH EXPORT START (V2) <<<")
print(f"Target: {OUTPUT_BVH}")

# Check Blender
blender_cmd = "blender" # Assume in PATH or alias

# Build Command
cmd = [
    blender_cmd,
    "--background",
    "--python", BLENDER_SCRIPT,
    "--",
    JSON_PATH,
    OUTPUT_BVH
]

print(f"Running: {' '.join(cmd)}")
try:
    subprocess.run(cmd, check=True)
    print(f"Success! Output: {OUTPUT_BVH}")
except subprocess.CalledProcessError as e:
    print(f"Error running Blender: {e}")
    sys.exit(1)
