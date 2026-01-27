
import bpy
import os
import sys
import json
try:
    import numpy as np
except ImportError:
    # Search in system site-packages (Colab/Ubuntu default)
    # Colab usually uses Python 3.10, 3.11, or 3.12
    for p in ["/usr/local/lib/python3.10/dist-packages", "/usr/local/lib/python3.11/dist-packages", "/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages"]:
        if os.path.exists(p) and p not in sys.path:
            sys.path.append(p)
    try:
        import numpy as np
    except ImportError:
        # Final fallback: Blender 3.0.1 might need python3-numpy from apt
        print("⚠ Warning: numpy not found. Please ensure 'python3-numpy' is installed via apt.")
        raise
from mathutils import Vector

def create_and_export_fbx_final(data, export_path):
    print(f"Starting Blender Humanoid Builder V2... Output: {export_path}")
    
    def fix_c(p):
        if p is None or (isinstance(p, (list, tuple, np.ndarray)) and (any(np.isnan(p)) or len(p) < 3)): 
            return Vector((0,0,0))
        # MHR (x, y, z) -> Blender (x, z, -y)
        # Note: Depending on MHR source, it might be meters.
        # Check scale?
        v = Vector((p[0], p[2], -p[1]))
        return v

    j_u = data.get("joints_mhr70", [])
    # Fallback if joints_mhr70 is inside 'pred_keypoints_3d'
    if not j_u and "pred_keypoints_3d" in data:
        j_u = data["pred_keypoints_3d"]
        
    verts = data.get("vertices", data.get("pred_vertices", []))
    faces = data.get("faces", [])
    weights = data.get("weights", [])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.0

    # --- 1. Mesh & Skinning ---
    mesh_obj = None
    if len(verts) > 0:
        print(f"Creating Mesh with {len(verts)} verts...")
        mesh_data = bpy.data.meshes.new("BodyMesh")
        mesh_obj = bpy.data.objects.new("BodyMesh", mesh_data)
        bpy.context.collection.objects.link(mesh_obj)
        
        # Convert Verts
        mesh_verts = [fix_c(v) for v in verts]
        
        # Validate Faces relative to Verts count
        num_verts = len(mesh_verts)
        valid_faces = []
        if len(faces) > 0:
            for f in faces:
                if max(f) < num_verts:
                    valid_faces.append(f)
        
        mesh_data.from_pydata(mesh_verts, [], valid_faces)
        mesh_data.update()
        
        # Auto Weights later or manual weights?
        # blender_app.py uses 'weights' from data if available.
        # If not, we will rely on Auto Weights.
        if len(weights) > 0:
            # Not implementing manual weights import here to keep it simple for generic mesh.
            # We will use Automatic Weights.
            pass

    # --- 2. Armature Generation ---
    print("Building Armature...")
    armature_data = bpy.data.armatures.new("Humanoid_Armature")
    rig = bpy.data.objects.new("Humanoid_Rig", armature_data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    eb = armature_data.edit_bones

    # Coordinate Variables (From blender_app.py)
    # Mapping indices: 
    # 0:Nose, 3:LEar, 4:REar, 9:LHip, 10:RHip, 69:Neck
    
    # Ensure j_u has data
    if len(j_u) < 70:
        print("Error: joints_mhr70 has insufficient data.")
        sys.exit(1)

    print(f"DEBUG: j_u[0] (Nose) raw: {j_u[0]}")
    print(f"DEBUG: j_u[9] (LeftHip) raw: {j_u[9]}")
    
    v_hips = (fix_c(j_u[9]) + fix_c(j_u[10])) / 2
    print(f"DEBUG: v_hips (Calculated): {v_hips}")
    
    v_neck = fix_c(j_u[69]) 
    v_head_base = (fix_c(j_u[3]) + fix_c(j_u[4])) / 2
    v_nose = fix_c(j_u[0])
    
    # Spine Interpolation
    # Spine Interpolation (Adjusted Ratios V4)
    # User feedback: UpperChest 35%, Chest 30%, Spine 35%.
    # Result: Spine 35%, Chest 30%, UpperChest 35%.
    v_spine0 = v_hips + (v_neck - v_hips) * 0.35 # Spine (Lumbar)
    v_spine1 = v_hips + (v_neck - v_hips) * 0.65 # Chest (Thoracic)
    v_spine2 = v_neck # UpperChest Top = Neck Base
    
    # Head Orientation
    f_fw = (v_nose - v_head_base).normalized() if (v_nose - v_head_base).length > 1e-6 else Vector((0,1,0))
    f_si = (fix_c(j_u[3]) - fix_c(j_u[4])).normalized() if (fix_c(j_u[3]) - fix_c(j_u[4])).length > 1e-6 else Vector((1,0,0))
    up_v = f_si.cross(f_fw).normalized()
    if up_v.z < 0: up_v = -up_v
    v_head_top = v_head_base + up_v * 0.15
    
    v_spine_head = v_hips + (v_spine0 - v_hips) * 0.4
    
    # Hips Orientation (Trusting AI's Root rotation)
    v_spine_dir = (v_spine0 - v_hips).normalized()
    if v_spine_dir.length < 1e-4: v_spine_dir = Vector((0, 0, 1))
    # Hips bone points towards Spine (Waist)
    v_hips_tail_rotated = v_hips + v_spine_dir * 0.1

    # Hands/Feet Ends
    l_h_end = (fix_c(j_u[62]) + fix_c(j_u[53])) / 2 
    r_h_end = (fix_c(j_u[41]) + fix_c(j_u[32])) / 2
    
    v_l_ankle = fix_c(j_u[13])
    v_r_ankle = fix_c(j_u[14])
    l_toe_t = (fix_c(j_u[15]) + fix_c(j_u[16])) / 2 
    r_toe_t = (fix_c(j_u[18]) + fix_c(j_u[19])) / 2
    
    l_f_end = v_l_ankle + (l_toe_t - v_l_ankle) * 0.5
    r_f_end = v_r_ankle + (r_toe_t - v_r_ankle) * 0.5

    # --- STRUCTURE DEFINITION (MHR Name -> Unity Name) ---
    struct = [
        # Hips: Points towards Spine. Head=Pelvis, Tail=Waist.
        (None, "Hips", v_hips, v_hips_tail_rotated, False),
        
        # Spine: Connects to Hips Tail (Waist).
        ("Hips", "Spine", v_hips_tail_rotated, v_spine0, True), 
        
        ("Spine", "Chest", v_spine0, v_spine1, True),
        ("Chest", "UpperChest", v_spine1, v_spine2, True),
        ("UpperChest", "Neck", v_neck, v_head_base, True), # Connect to Neck
        ("Neck", "Head", v_head_base, v_head_top, True),
        ("Head", "Head_end", v_head_top, v_head_top + up_v * 0.05, True), 
        
        # Arms
        ("UpperChest", "LeftShoulder", (v_neck+fix_c(j_u[5]))/2, fix_c(j_u[5]), False),
        ("LeftShoulder", "LeftUpperArm", fix_c(j_u[5]), fix_c(j_u[7]), True),
        ("LeftUpperArm", "LeftLowerArm", fix_c(j_u[7]), fix_c(j_u[62]), True),
        ("LeftLowerArm", "LeftHand", fix_c(j_u[62]), l_h_end, True),
        
        ("UpperChest", "RightShoulder", (v_neck+fix_c(j_u[6]))/2, fix_c(j_u[6]), False),
        ("RightShoulder", "RightUpperArm", fix_c(j_u[6]), fix_c(j_u[8]), True),
        ("RightUpperArm", "RightLowerArm", fix_c(j_u[8]), fix_c(j_u[41]), True),
        ("RightLowerArm", "RightHand", fix_c(j_u[41]), r_h_end, True),
        
        # Legs - Parent to Hips (Visual link to Hips Tail)
        ("Hips", "LeftUpperLeg", fix_c(j_u[9]), fix_c(j_u[11]), False),
        ("LeftUpperLeg", "LeftLowerLeg", fix_c(j_u[11]), v_l_ankle, True),
        ("LeftLowerLeg", "LeftFoot", v_l_ankle, l_f_end, True),
        ("LeftFoot", "LeftToes", l_f_end, l_toe_t, True),
        ("LeftToes", "LeftToes_end", l_toe_t, l_toe_t + (l_toe_t-l_f_end).normalized()*0.02 if (l_toe_t-l_f_end).length>1e-6 else l_toe_t+Vector((0,0.02,0)), True),
        
        ("Hips", "RightUpperLeg", fix_c(j_u[10]), fix_c(j_u[12]), False),
        ("RightUpperLeg", "RightLowerLeg", fix_c(j_u[12]), v_r_ankle, True),
        ("RightLowerLeg", "RightFoot", v_r_ankle, r_f_end, True),
        ("RightFoot", "RightToes", r_f_end, r_toe_t, True),
        ("RightToes", "RightToes_end", r_toe_t, r_toe_t + (r_toe_t-r_f_end).normalized()*0.02 if (r_toe_t-r_f_end).length>1e-6 else r_toe_t+Vector((0,0.02,0)), True),
    ]

    for p_name, b_name, h, t, use_conn in struct:
        b = eb.new(b_name)
        b.head, b.tail = h, t
        if p_name: b.parent, b.use_connect = eb[p_name], use_conn

    # --- Fingers (Simplified mapping) ---
    def add_f(h_bone, wrist_idx, indices, prefix):
        # Indices: [Proximal, Intermediate, Distal, End?] 
        # Using blender_app indices: [45, 44, 43, 42] -> P, I, D, Tip
        
        # Unity: Proximal, Intermediate, Distal
        parts = ["Proximal", "Intermediate", "Distal"]
        parent_bone = eb[h_bone]
        
        # Connect Wrist -> First Joint (Metacarpal area)
        # Unity usually treats IndexProximal as the knuckle.
        # So Wrist -> Knuckle is implicit or Metacarpal.
        # We will create Metacarpal for all to be safe (Unity can ignore).
        
        # Metacarpal
        meta_name = f"{prefix}Metacarpal" # Optional in Unity?
        b_meta = eb.new(meta_name)
        b_meta.head = fix_c(j_u[wrist_idx])
        b_meta.tail = fix_c(j_u[indices[0]])
        b_meta.parent = parent_bone
        b_meta.use_connect = False
        
        prev_bone = b_meta
        
        for i in range(len(parts)):
            if i >= len(indices) - 1: break
            name = f"{prefix}{parts[i]}"
            b = eb.new(name)
            b.head = fix_c(j_u[indices[i]])
            b.tail = fix_c(j_u[indices[i+1]])
            b.parent = prev_bone
            b.use_connect = True
            prev_bone = b
            
    # Mappings
    # Thumb: 45,44,43,42. 45=CMC?
    add_f("LeftHand", 62, [45, 44, 43, 42], "LeftThumb")
    add_f("LeftHand", 62, [49, 48, 47, 46], "LeftIndex")
    add_f("LeftHand", 62, [53, 52, 51, 50], "LeftMiddle")
    add_f("LeftHand", 62, [57, 56, 55, 54], "LeftRing")
    add_f("LeftHand", 62, [61, 60, 59, 58], "LeftLittle")
    
    add_f("RightHand", 41, [24, 23, 22, 21], "RightThumb")
    add_f("RightHand", 41, [28, 27, 26, 25], "RightIndex")
    add_f("RightHand", 41, [32, 31, 30, 29], "RightMiddle")
    add_f("RightHand", 41, [36, 35, 34, 33], "RightRing")
    add_f("RightHand", 41, [40, 39, 38, 37], "RightLittle")

    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Parent Mesh to Armature
    if mesh_obj:
        print("Parenting Mesh to Armature (Automatic Weights)...")
        bpy.ops.object.select_all(action='DESELECT')
        armature_data.pose_position = 'REST'
        rig.select_set(True)
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')

    # Export
    # Use generic FBX settings
    bpy.ops.export_scene.fbx(
        filepath=export_path, 
        use_selection=True, 
        add_leaf_bones=False, 
        axis_forward='-Z', 
        axis_up='Y',
        bake_anim=False
    )
    print(f"FBX Success: {export_path}")

if __name__ == "__main__":
    # Args: -- input_json output_fbx
    # We expect raw arguments or standard sys.argv
    # Let's robustly find args
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        
    if len(args) < 2:
        print("Usage: blender --python script.py -- input.json output.fbx")
        sys.exit(1)
        
    rp_path = args[0]
    out_path = args[1]
    
    with open(rp_path,'r') as f: data = json.load(f)
    create_and_export_fbx_final(data, out_path)
