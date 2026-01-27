import numpy as np
import json
import os
import sys
import subprocess

# Determine paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
LIB_DIR = os.path.join(SCRIPT_DIR, "lib")

# ------------------------------------------------------------------------
# JSON ENCODER FOR NUMPY
# ------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# -----------------------------------------------------------------------------
# CONSTANTS: UNITY HUMANOID HIERARCHY & MAPPING
# -----------------------------------------------------------------------------

# Unity Bone -> Parent
# Used to build the skeleton structure in JSON
HIERARCHY = {
    "Hips": None,
    "Spine": "Hips",
    "Chest": "Spine",
    "UpperChest": "Chest",
    "Neck": "UpperChest",
    "Head": "Neck",
    "Head_end": "Head",
    
    "LeftShoulder": "UpperChest",
    "LeftUpperArm": "LeftShoulder",
    "LeftLowerArm": "LeftUpperArm",
    "LeftHand": "LeftLowerArm",
    
    "RightShoulder": "UpperChest",
    "RightUpperArm": "RightShoulder",
    "RightLowerArm": "RightUpperArm",
    "RightHand": "RightLowerArm",
    
    "LeftUpperLeg": "Hips",
    "LeftLowerLeg": "LeftUpperLeg",
    "LeftFoot": "LeftLowerLeg",
    "LeftToes": "LeftFoot",
    "LeftToes_end": "LeftToes",
    
    "RightUpperLeg": "Hips",
    "RightLowerLeg": "RightUpperLeg",
    "RightFoot": "RightLowerLeg",
    "RightToes": "RightFoot",
    "RightToes_end": "RightToes",
}

# Add Fingers
for side in ["Left", "Right"]:
    for f in ["Thumb", "Index", "Middle", "Ring", "Little"]:
        # Proximal -> Intermediate -> Distal -> End
        HIERARCHY[f"{side}{f}Proximal"] = f"{side}Hand"
        HIERARCHY[f"{side}{f}Intermediate"] = f"{side}{f}Proximal"
        HIERARCHY[f"{side}{f}Distal"] = f"{side}{f}Intermediate"
        HIERARCHY[f"{side}{f}Distal_end"] = f"{side}{f}Distal"

# MHR-127 Joint Names (Ordered List from output_3.npy)
MHR_JOINT_NAMES = [
    'body_world', 'root', 'l_upleg', 'l_lowleg', 'l_foot', 'l_talocrural', 'l_subtalar', 
    'l_transversetarsal', 'l_ball', 'l_lowleg_twist1_proc', 'l_lowleg_twist2_proc', 'l_lowleg_twist3_proc', 
    'l_lowleg_twist4_proc', 'l_upleg_twist0_proc', 'l_upleg_twist1_proc', 'l_upleg_twist2_proc', 'l_upleg_twist3_proc', 
    'l_upleg_twist4_proc', 'r_upleg', 'r_lowleg', 'r_foot', 'r_talocrural', 'r_subtalar', 'r_transversetarsal', 'r_ball', 
    'r_lowleg_twist1_proc', 'r_lowleg_twist2_proc', 'r_lowleg_twist3_proc', 'r_lowleg_twist4_proc', 'r_upleg_twist0_proc', 
    'r_upleg_twist1_proc', 'r_upleg_twist2_proc', 'r_upleg_twist3_proc', 'r_upleg_twist4_proc', 'c_spine0', 'c_spine1', 
    'c_spine2', 'c_spine3', 'r_clavicle', 'r_uparm', 'r_lowarm', 'r_wrist_twist', 'r_wrist', 'r_pinky0', 'r_pinky1', 'r_pinky2', 
    'r_pinky3', 'r_pinky_null', 'r_ring1', 'r_ring2', 'r_ring3', 'r_ring_null', 'r_middle1', 'r_middle2', 'r_middle3', 
    'r_middle_null', 'r_index1', 'r_index2', 'r_index3', 'r_index_null', 'r_thumb0', 'r_thumb1', 'r_thumb2', 'r_thumb3', 
    'r_thumb_null', 'r_lowarm_twist1_proc', 'r_lowarm_twist2_proc', 'r_lowarm_twist3_proc', 'r_lowarm_twist4_proc', 
    'r_uparm_twist0_proc', 'r_uparm_twist1_proc', 'r_uparm_twist2_proc', 'r_uparm_twist3_proc', 'r_uparm_twist4_proc', 
    'l_clavicle', 'l_uparm', 'l_lowarm', 'l_wrist_twist', 'l_wrist', 'l_pinky0', 'l_pinky1', 'l_pinky2', 'l_pinky3', 
    'l_pinky_null', 'l_ring1', 'l_ring2', 'l_ring3', 'l_ring_null', 'l_middle1', 'l_middle2', 'l_middle3', 'l_middle_null', 
    'l_index1', 'l_index2', 'l_index3', 'l_index_null', 'l_thumb0', 'l_thumb1', 'l_thumb2', 'l_thumb3', 'l_thumb_null', 
    'l_lowarm_twist1_proc', 'l_lowarm_twist2_proc', 'l_lowarm_twist3_proc', 'l_lowarm_twist4_proc', 'l_uparm_twist0_proc', 
    'l_uparm_twist1_proc', 'l_uparm_twist2_proc', 'l_uparm_twist3_proc', 'l_uparm_twist4_proc', 'c_neck', 'c_neck_twist1_proc', 
    'c_neck_twist0_proc', 'c_head', 'c_jaw', 'c_teeth', 'c_jaw_null', 'c_tongue0', 'c_tongue1', 'c_tongue2', 'c_tongue3', 
    'c_tongue4', 'r_eye', 'r_eye_null', 'l_eye', 'l_eye_null', 'c_head_null'
]

MHR_NAME_TO_IDX = {name: i for i, name in enumerate(MHR_JOINT_NAMES)}

# Unity Name -> MHR Name
UNITY_TO_MHR = {
    "Hips": "root",
    "Spine": "c_spine0", "Chest": "c_spine1", "UpperChest": "c_spine2",
    "Neck": "c_neck", "Head": "c_head", "Head_end": "c_head_null",
    "LeftShoulder": "l_clavicle", "LeftUpperArm": "l_uparm", "LeftLowerArm": "l_lowarm", "LeftHand": "l_wrist", 
    "RightShoulder": "r_clavicle", "RightUpperArm": "r_uparm", "RightLowerArm": "r_lowarm", "RightHand": "r_wrist",
    "LeftUpperLeg": "l_upleg", "LeftLowerLeg": "l_lowleg", "LeftFoot": "l_foot", "LeftToes": "l_ball", "LeftToes_end": "l_ball", # Approx end
    "RightUpperLeg": "r_upleg", "RightLowerLeg": "r_lowleg", "RightFoot": "r_foot", "RightToes": "r_ball", "RightToes_end": "r_ball",
}
# Fingers
for side, prefix in [("l", "Left"), ("r", "Right")]:
    for f_u, f_m in [("Thumb", "thumb"), ("Index", "index"), ("Middle", "middle"), ("Ring", "ring"), ("Little", "pinky")]:
        for i, part in enumerate(["Proximal", "Intermediate", "Distal", "Distal_end"]):
             # thumb0, thumb1... or index1, index2...
             # MHR: thumb0,1,2,3(null). Index1,2,3,null.
             # Unity: Proximal, Intermediate, Distal, End.
             
             # Thumb Mapping: P=0, I=1, D=2, E=3
             # Finger Mapping: P=1, I=2, D=3, E=null
             
             idx = i
             if f_m != "thumb": idx += 1 # Index starts at 1
             
             m_name = f"{side}_{f_m}{idx}"
             if idx > 3: m_name = f"{side}_{f_m}_null" # Fallback
             
             u_name = f"{prefix}{f_u}{part}"
             UNITY_TO_MHR[u_name] = m_name


def get_windows_path(wsl_path):
    if not wsl_path: return wsl_path
    if "WSL_DISTRO_NAME" not in os.environ: return wsl_path
    try:
        # Use wslpath -m (Format with forward slashes)
        # return strict windows path? -m creates //wsl.localhost/...
        # Blender needs this to access network share
        result = subprocess.run(['wslpath', '-m', wsl_path], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"Warning: Failed to convert WSL path {wsl_path}: {e}")
        return wsl_path

def main():
    print(">>> GENERATE GENERIC MODEL START (V2) <<<")
    
    npy_path = os.path.join(PARENT_DIR, "outputs/output_3.npy")
    if not os.path.exists(npy_path):
        print(f"Error: {npy_path} not found.")
        return

    data = np.load(npy_path, allow_pickle=True).item()
    # --- REVISED DATA EXTRACTION (Fix Zeros & Offset) ---
    # Prioritize pred_keypoints_3d because joints_mhr70 has corrupted Head (Zeros)
    
    frame_idx = 0
    
    # 1. Get Vertices
    raw_verts = data['pred_vertices']
    if len(raw_verts.shape) == 3: verts = raw_verts[frame_idx]
    else: verts = raw_verts
    if isinstance(verts, np.ndarray): verts = verts.reshape(-1, 3)
    
    # 2. Get Joints
    # Use pred_keypoints_3d (127 points) as primary if available
    if "pred_keypoints_3d" in data:
        j_raw = data["pred_keypoints_3d"]
        print("Using pred_keypoints_3d (Primary source)")
    elif "joints_mhr70" in data:
        j_raw = data["joints_mhr70"]
        print("Using joints_mhr70 (Secondary source)")
    else:
        print("Error: No joint data found.")
        return

    if len(j_raw.shape) == 3: joints = j_raw[frame_idx]
    else: joints = j_raw
    if isinstance(joints, np.ndarray): joints = joints.reshape(-1, 3)
    
    # 3. Validation
    # Check if Nose (0) is 0. If so, try to fix or abort?
    # If pred_keypoints_3d is used, usually index 0 is Nose and valid.
    print(f"DEBUG: Joint[0] (Nose) Raw: {joints[0]}")
    
    # 4. CENTERING (Fix 5m Offset)
    # Both Verts and Joints are in Camera Space (Z ~ 5.0m).
    # We want the model at Origin (0,0,0).
    # Calculate Centroid of VERTICES
    centroid = np.mean(verts, axis=0)
    print(f"DEBUG: Vertex Centroid: {centroid}")
    
    # Subtract Centroid from BOTH
    verts = verts - centroid
    joints = joints - centroid
    
    print(f"DEBUG: Centered Joint[0]: {joints[0]}")
    print(f"DEBUG: Centered Vert[0]: {verts[0]}")
    
    # Faces
    faces = data.get('faces', [])
    if hasattr(faces, 'tolist'): faces = faces.tolist()

    # 2. Package for Blender
    payload = {
        "joints_mhr70": joints.tolist(),
        "vertices": verts.tolist(),
        "faces": faces
    }
    
    temp_dir = os.path.join(PARENT_DIR, "outputs/temp_generic")
    os.makedirs(temp_dir, exist_ok=True)
    json_path = os.path.join(temp_dir, "v2_payload.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, cls=NumpyEncoder)
        
    print(f"Saved payload to {json_path}")
    
    # 3. Run Blender
    import shutil
    blender_exe = None
    is_windows_blender = False
    
    if shutil.which("blender") is not None:
        blender_exe = "blender"
    
    if blender_exe is None:
        win_paths = [
            "/mnt/c/Program Files/Blender Foundation/Blender 4.3/blender.exe",
            "/mnt/c/Program Files/Blender Foundation/Blender 4.2/blender.exe",
            "/mnt/c/Program Files/Blender Foundation/Blender 3.6/blender.exe",
            "/mnt/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
        ]
        for p in win_paths:
            if os.path.exists(p):
                blender_exe = p
                is_windows_blender = True
                break
                
    if blender_exe is None:
        print("Error: Blender not found.")
        return

    script_path = os.path.join(LIB_DIR, "blender_humanoid_builder_v2.py")
    output_fbx = os.path.join(PARENT_DIR, "generic_humanoid.fbx")
    
    def smart_resolve(p):
        abs_p = os.path.abspath(p)
        if is_windows_blender: return get_windows_path(abs_p)
        return abs_p

    r_script = smart_resolve(script_path)
    r_json = smart_resolve(json_path)
    r_out = smart_resolve(output_fbx)
    
    cmd = [
        blender_exe, 
        "--background", 
        "--python", r_script, 
        "--", 
        r_json, 
        r_out
    ]
    
    print(f"Running Blender: {blender_exe}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Blender Error:")
        print(result.stderr)
    else:
        if os.path.exists(output_fbx):
            print(f"Success! Output: {output_fbx}")
        else:
            print("Blender finished but file missing.")

if __name__ == "__main__":
    main()
