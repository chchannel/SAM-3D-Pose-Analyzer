import bpy
import os
import sys
import math

def convert_fbx_to_bvh_crysta(fbx_path, export_path):
    print(f"Converting FBX to BVH for Clip Studio Paint...")
    print(f"Source: {fbx_path}")
    print(f"Target: {export_path}")
    
    # 1. Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # 2. Import the finalized FBX
    if not os.path.exists(fbx_path):
        print(f"Error: {fbx_path} not found.")
        return
        
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    
    # 3. Find Armature
    rig = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            rig = obj
            break
            
    if not rig:
        print("Error: No Armature found in FBX.")
        return
        
    # 4. Apply 90-degree X-rotation to make it upright (-Y -> Z)
    # User Request: "Blender -y side becomes z"
    # Rotate 90 degrees around X axis
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    
    # --- Head/Neck Alignment Fix ---
    print("Re-aligning Head/Neck to Spine direction...")
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    
    # Names from blender_humanoid_builder_v2.py
    b_ups = eb.get("UpperChest")
    b_neck = eb.get("Neck")
    b_head = eb.get("Head")
    
    if b_ups and b_neck and b_head:
        # Calculate spine direction from UpperChest
        spine_vec = (b_ups.tail - b_ups.head).normalized()
        if spine_vec.length > 1e-6:
            # Re-orient Neck and Head to follow this vector
            # We keep the chain connected
            n_len = (b_neck.tail - b_neck.head).length
            h_len = (b_head.tail - b_head.head).length
            
            b_neck.tail = b_neck.head + spine_vec * n_len
            b_head.tail = b_head.head + spine_vec * h_len
            print("  ✅ Head/Neck aligned to Spine.")
    else:
        print("  ⚠ Warning: Could not find UpperChest, Neck or Head.")
        
    bpy.ops.object.mode_set(mode='OBJECT')
    
    print("Rotating world context: Blender -Y -> World Z (Correcting Upside-Down)...")
    rig.rotation_euler = (math.radians(-90), 0, 0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # 5. Export to BVH
    # Force single frame 1
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 1
    
    bpy.ops.export_anim.bvh(
        filepath=export_path,
        check_existing=False,
        global_scale=1.0,
        frame_start=1,
        frame_end=1,
        root_transform_only=False
    )
    print(f"BVH Export Successful: {export_path}")

if __name__ == "__main__":
    # 引数取得: blender --python script.py -- [SOURCE_FBX] [OUTPUT_BVH]
    try:
        args = sys.argv[sys.argv.index("--") + 1:]
        if len(args) < 2:
            print("Usage: blender --background --python script.py -- <source_fbx> <output_bvh>")
            sys.exit(1)
            
        fbx_file = args[0]
        export_file = args[1]
        convert_fbx_to_bvh_crysta(fbx_file, export_file)
    except (ValueError, IndexError):
        print("Error: Invalid arguments.")
        sys.exit(1)
