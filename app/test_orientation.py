import numpy as np
import json
import os
import subprocess
import sys

# パス設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
NPY_PATH = os.path.join(OUTPUT_DIR, "output_0.npy")
JSON_PATH = os.path.join(OUTPUT_DIR, "tjson_test_rotation.json")
FBX_PATH = os.path.join(OUTPUT_DIR, "output_test_rotation.fbx")
BLENDER_SCRIPT = os.path.join(BASE_DIR, "convert", "lib", "blender_humanoid_builder_test_rotation.py")

def main():
    if not os.path.exists(NPY_PATH):
        print(f"❌ Error: {NPY_PATH} not found. Please run an inference first.")
        return

    print(f"📖 Loading {NPY_PATH}...")
    data = np.load(NPY_PATH, allow_pickle=True).item()

    # Blender スクリプトに渡すための JSON 作成
    # predict_worker.py の Step 4 と同様の形式
    test_json = {
        "vertices": data["pred_vertices"].tolist(),
        "faces": data["faces"].tolist(),
        "joints_mhr70": data["pred_keypoints_3d"].tolist()
    }

    print(f"📝 Writing {JSON_PATH}...")
    with open(JSON_PATH, "w") as f:
        json.dump(test_json, f)

    print(f"🚀 Running Blender with rotation-aware script...")
    cmd = [
        "blender", "--background", "--python", BLENDER_SCRIPT,
        "--", JSON_PATH, FBX_PATH
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Success! FBX generated at: {FBX_PATH}")
        print("You can now open this FBX in Blender to check if the character is rotated correctly.")
    else:
        print(f"❌ Blender Error:\n{result.stderr}")
        print(result.stdout)

if __name__ == "__main__":
    main()
