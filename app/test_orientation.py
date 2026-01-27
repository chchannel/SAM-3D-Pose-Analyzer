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

    # 出力パス設定
    FBX_PATH = os.path.join(OUTPUT_DIR, "output_test_rotation.fbx")
    BVH_LEGACY = os.path.join(OUTPUT_DIR, "test_legacy_fix.bvh")
    BVH_MODERN_STD = os.path.join(OUTPUT_DIR, "test_modern_standard.bvh")
    BVH_MODERN_FIX = os.path.join(OUTPUT_DIR, "test_modern_head_fix.bvh")

    LIB_DIR = os.path.join(BASE_DIR, "convert", "lib")
    BLENDER_BUILDER = os.path.join(LIB_DIR, "blender_humanoid_builder_test_rotation.py")
    BLENDER_BVH_STD = os.path.join(LIB_DIR, "blender_bvh_crysta.py")
    BLENDER_BVH_FIX = os.path.join(LIB_DIR, "blender_bvh_crysta_head_fix.py")
    LEGACY_CONVERTER = os.path.join(BASE_DIR, "convert", "convert_to_bvh.py")

    print(f"🚀 [1/4] Generating FBX (Head-to-Spine fixed builder)...")
    cmd_fbx = ["blender", "--background", "--python", BLENDER_BUILDER, "--", JSON_PATH, FBX_PATH]
    subprocess.run(cmd_fbx, capture_output=True)

    print(f"🚀 [2/4] Generating BVH from FBX (Standard Converter)...")
    cmd_bvh_std = ["blender", "--background", "--python", BLENDER_BVH_STD, "--", FBX_PATH, BVH_MODERN_STD]
    subprocess.run(cmd_bvh_std, capture_output=True)

    print(f"🚀 [3/4] Generating BVH from FBX (Post-Import Head Fix)...")
    cmd_bvh_fix = ["blender", "--background", "--python", BLENDER_BVH_FIX, "--", FBX_PATH, BVH_MODERN_FIX]
    subprocess.run(cmd_bvh_fix, capture_output=True)

    print(f"🚀 [4/4] Generating Legacy BVH (JSON direct converter)...")
    cmd_legacy = [sys.executable, LEGACY_CONVERTER, NPY_PATH, BVH_LEGACY]
    subprocess.run(cmd_legacy, capture_output=True)

    print(f"\n✅ All tests completed! Check these files in {OUTPUT_DIR}:")
    print(f"  1. {os.path.basename(FBX_PATH)} (FBX w/ rotation & head fix)")
    print(f"  2. {os.path.basename(BVH_LEGACY)} (Legacy JSON->BVH fix)")
    print(f"  3. {os.path.basename(BVH_MODERN_STD)} (Modern FBX->BVH)")
    print(f"  4. {os.path.basename(BVH_MODERN_FIX)} (Modern FBX->BVH w/ post-import fix)")

if __name__ == "__main__":
    main()
