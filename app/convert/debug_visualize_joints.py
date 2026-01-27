
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_joints(npy_path, output_image):
    data = np.load(npy_path, allow_pickle=True).item()
    raw_joints = data['pred_joint_coords']
    print(f"Raw joints shape: {raw_joints.shape}")
    
    if len(raw_joints.shape) == 3:
        joints = raw_joints[0]
    else:
        joints = raw_joints
    
    print(f"Processing joints shape: {joints.shape}")
    
    # Filter valid points (remove close to 0 or too large)
    valid_indices = []
    valid_coords = []
    for i in range(min(70, joints.shape[0])): # Check first 70
        pt = joints[i]
        if np.all(np.abs(pt) < 100) and not np.all(pt == 0):
            valid_indices.append(i)
            valid_coords.append(pt)
            
    valid_coords = np.array(valid_coords)
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(valid_coords[:, 0], valid_coords[:, 2], valid_coords[:, 1], c='blue', marker='o') # Swap Y-Z for visualization if needed
    
    # Label points
    for i, idx in enumerate(valid_indices):
        x, y, z = valid_coords[i]
        # Label with Index
        ax.text(x, z, y, str(idx), fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Z') # Visual Y
    ax.set_zlabel('Y') # Visual Z
    ax.set_title(f"Joint Indices 0-69 (Shape: {joints.shape})")
    
    # Set equal aspect ratio
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([valid_coords[:, 0].max()-valid_coords[:, 0].min(), 
                          valid_coords[:, 2].max()-valid_coords[:, 2].min(), 
                          valid_coords[:, 1].max()-valid_coords[:, 1].min()]).max() / 2.0

    mid_x = (valid_coords[:, 0].max()+valid_coords[:, 0].min()) * 0.5
    mid_y = (valid_coords[:, 2].max()+valid_coords[:, 2].min()) * 0.5 # Swapped
    mid_z = (valid_coords[:, 1].max()+valid_coords[:, 1].min()) * 0.5 # Swapped
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.savefig(output_image)
    print(f"Saved visualization to {output_image}")
    
    # Text Analysis
    print("\n--- Checking MHR Standard Indices (Hypothesis) ---")
    mhr_standard = {
        0: "body_world",
        1: "root",
        2: "l_upleg (Hip?)",
        3: "l_lowleg (Knee?)",
        4: "l_foot (Ankle?)",
        18: "r_upleg (Hip?)",
        19: "r_lowleg",
        34: "c_spine0",
        37: "c_spine3",
        101: "c_neck",
        103: "c_head"
    }
    
    for idx, name in mhr_standard.items():
        if idx < len(joints):
            p = joints[idx]
            print(f"{idx} ({name}): {p}")
            
    # Check hierarchy Y/Z consistency
    print("\nHierarchy Check (Standard Order):")
    if 103 in mhr_standard and 101 in mhr_standard and 34 in mhr_standard:
        head = joints[103]
        neck = joints[101]
        spine = joints[34]
        hip = joints[2]
        
        print(f"Head (103): {head}")
        print(f"Neck (101): {neck}")
        print(f"Spine (34): {spine}")
        print(f"L_Hip (2) : {hip}")
        
        print(f"Head -> Neck: {neck - head}")
        print(f"Neck -> Spine: {spine - neck}")
        print(f"Spine -> Hip: {hip - spine}")


if __name__ == "__main__":
    npy_path = "outputs/20260121_202557_output_2.npy"
    if not os.path.exists(npy_path):
        # Fallback to latest
        import glob
        files = glob.glob("outputs/*.npy")
        if files:
            npy_path = max(files, key=os.path.getctime)
    
    visualize_joints(npy_path, "debug_joints_viz.png")
