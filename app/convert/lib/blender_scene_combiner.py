
import bpy
import os
import sys
import json
import numpy as np
from mathutils import Vector

def combine_and_export_glb(json_paths, export_path):
    print(f"Combining {len(json_paths)} persons into one GLB...")
    
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.0

    def fix_c(p):
        # MHR (x, y, z) -> Blender (x, z, -y)
        return Vector((p[0], p[2], -p[1]))

    for i, path in enumerate(json_paths):
        if not os.path.exists(path): continue
        with open(path, 'r') as f:
            data = json.load(f)
        
        verts = data.get("vertices", data.get("pred_vertices", []))
        faces = data.get("faces", [])
        
        if len(verts) == 0: continue
        
        mesh_data = bpy.data.meshes.new(f"Person_{i}_Mesh")
        mesh_obj = bpy.data.objects.new(f"Person_{i}", mesh_data)
        bpy.context.collection.objects.link(mesh_obj)
        
        mesh_verts = [fix_c(v) for v in verts]
        mesh_data.from_pydata(mesh_verts, [], faces)
        mesh_data.update()

    # Export as GLB
    bpy.ops.export_scene.gltf(
        filepath=export_path,
        export_format='GLB',
        use_selection=False
    )
    print(f"Combined GLB exported: {export_path}")

if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(args) < 2:
        print("Usage: blender --background --python script.py -- output.glb input1.json input2.json ...")
        sys.exit(1)
        
    out_p = args[0]
    in_ps = args[1:]
    combine_and_export_glb(in_ps, out_p)
