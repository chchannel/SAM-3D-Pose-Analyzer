import sys
import os
import torch
import torch.serialization
import torch.hub

# --- PyTorch 2.4+ Compatibility Patches ---
_compatible_types = []
for s in ['UInt16Storage', 'UInt32Storage', 'UInt64Storage', 'Int64Storage', 'BoolStorage', 'BFloat16Storage', 'Float8_e4m3fnStorage', 'Float8_e5m2Storage']:
    if not hasattr(torch, s):
        c = type(s, (torch.UntypedStorage,), {})
        setattr(torch, s, c)
        _compatible_types.append(c)
    if hasattr(torch, 'storage') and not hasattr(torch.storage, s):
        setattr(torch.storage, s, getattr(torch, s))
try:
    torch.serialization.add_safe_globals(_compatible_types)
except Exception: pass

_orig_torch_load = torch.serialization.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.serialization.load = _patched_torch_load
torch.load = _patched_torch_load

_orig_hub_load = torch.hub.load
def _patched_hub_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_hub_load(*args, **kwargs)
torch.hub.load = _patched_hub_load

_orig_url_load = torch.hub.load_state_dict_from_url
def _patched_url_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_url_load(*args, **kwargs)
torch.hub.load_state_dict_from_url = _patched_url_load

for m_name, m in list(sys.modules.items()):
    if m:
        if hasattr(m, 'load') and (getattr(m, 'load') is _orig_torch_load):
            try: setattr(m, 'load', _patched_torch_load)
            except Exception: pass
        if hasattr(m, 'load_state_dict_from_url') and (getattr(m, 'load_state_dict_from_url') is _orig_url_load):
            try: setattr(m, 'load_state_dict_from_url', _patched_url_load)
            except Exception: pass
# -------------------------------------------------------------------------------------

import torch.nn as nn
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import subprocess
import time
import argparse
import gc
import shutil
import PIL.Image
import PIL.ImageOps

# ==========================================
# 🌍 パス解決 (ポータブル構成)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug_masks")
VIS_OUTPUT = os.path.join(BASE_DIR, "last_inference_vis.jpg")

REPOS_ROOT = os.path.join(PROJECT_ROOT, "repos")
SAM3D_ROOT = os.path.join(REPOS_ROOT, "sam-3d-body")
SAM3_ROOT = os.path.join(REPOS_ROOT, "sam3")
MOGE_ROOT = os.path.join(REPOS_ROOT, "MoGe")
# detectron2_repo が存在する場合はパスに追加
DETECTRON2_REPO = os.path.join(REPOS_ROOT, "detectron2_repo")
DETECTRON2_STUB = os.path.join(BASE_DIR, "detectron2_stub")

WEIGHTS_ROOT = os.path.join(PROJECT_ROOT, "weights", "body")
SAM3DB_CKPT = os.path.join(WEIGHTS_ROOT, "model.ckpt")
MHR_MODEL_PT = os.path.join(WEIGHTS_ROOT, "assets", "mhr_model.pt")
SAM3_CKPT = os.path.join(WEIGHTS_ROOT, "model.pt")

# 優先度の高い順にパスを追加 (後から insert(0) すると優先度が高くなる)
# detectron2_repo/tools と sam-3d-body/tools の競合を避けるため、sam-3d-body を優先
for p in [DETECTRON2_REPO, DETECTRON2_STUB, MOGE_ROOT, SAM3_ROOT, SAM3D_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# (旧式のカスタムローダーは HumanDetector への移行に伴い削除)

def cleanup_outputs():
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            p = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(p): os.remove(p)
            elif os.path.isdir(p): shutil.rmtree(p)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

def clear_memory():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--min_area", type=int, default=1000)
    parser.add_argument("--text_prompt", type=str, default="person")
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--sam3_only", action="store_true")
    parser.add_argument("--target_ids", type=str, default="")
    parser.add_argument("--use_moge", action="store_true")
    parser.add_argument("--clear_mem", action="store_true")
    parser.add_argument("--detector_name", type=str, default="sam3")
    parser.add_argument("--inference_type", type=str, default="full")
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--box_scale", type=float, default=1.2)
    parser.add_argument("--nms_thr", type=float, default=0.3)
    args = parser.parse_args()

    time_start = time.time()
    cleanup_outputs(); device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 画像読み込み (EXIF回転対応 & 自動リサイズ)
    try:
        pil_img = PIL.Image.open(args.image_path)
        pil_img = PIL.ImageOps.exif_transpose(pil_img)
        # OOM回避: 最大2048pxに縮小
        if max(pil_img.size) > 2048:
            print(f"📏 Resizing image from {pil_img.size} to max 2048px...")
            pil_img.thumbnail((2048, 2048), PIL.Image.LANCZOS)
        
        # --- ROBUST TRANSPARENCY HANDLING (Force White Background) ---
        rgba = pil_img.convert("RGBA")
        canvas = PIL.Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        # 透過があれば白背景の上に合成、不透明ならそのまま上書きされる
        pil_img = PIL.Image.alpha_composite(canvas, rgba).convert("RGB")
        # -------------------------------------------------------------
            
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"❌ ERROR: Failed to load image {args.image_path}: {e}")
        sys.exit(1)

    # [Step 1] Detection
    print(f"--- [Step 1] Detection using '{args.detector_name}' (prompt: '{args.text_prompt}') ---")
    from tools.build_detector import HumanDetector
    
    try:
        # ラッパーとしての設計: 
        # 原則的に sam-3d-body 公式の HumanDetector を使用しますが、
        # 公式コードが HF (ネット) 固定である SAM3 についてのみ、
        # ユーザーがローカルに sam3.pt を置いている場合に限り、読み込みを差し替えてオフライン動作を支援します。
        
        det_path = ""
        if args.detector_name == "vitdet":
            det_path = WEIGHTS_ROOT

        raw_sam3_output = None # SAM3の生出力を保持してマスクを取り出す
        if args.detector_name == "sam3":
            local_sam3 = os.path.join(WEIGHTS_ROOT, "sam3.pt")
            if os.path.exists(local_sam3):
                print(f"📦 Local SAM3 checkpoint found. Loading...")
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
                m_s3 = build_sam3_image_model(checkpoint_path=local_sam3, load_from_HF=False, device=device)
                
                # HumanDetectorを介さず直接構成（他モデルの二重読み込みを完全に防止）
                class SimpleDetector: pass
                detector = SimpleDetector()
                detector.device = device
                detector.detector = m_s3
                detector.processor = Sam3Processor(m_s3, device=device)
            else:
                print(f"  -> Initializing SAM3 (Online mode)...")
                detector = HumanDetector(name="sam3", device=device)
            
            # 生の出力を取得
            print(f"  -> Running SAM3 inference...")
            img_rgb_pil = PIL.Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            inference_state = detector.processor.set_image(img_rgb_pil)
            raw_sam3_output = detector.processor.set_text_prompt(state=inference_state, prompt=args.text_prompt)
            boxes = raw_sam3_output["boxes"].cpu().numpy()
            scores = raw_sam3_output["scores"].cpu().numpy()
            # スコアでフィルタリング
            keep = scores > args.conf_threshold
            boxes = boxes[keep]
            # HumanDetector.sam3_run が行う拡大処理を再現 (引数から反映)
            scale = args.box_scale
            enlarged_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = (x2 - x1) * scale, (y2 - y1) * scale
                enlarged_boxes.append([max(cx - w / 2, 0), max(cy - h / 2, 0), min(cx + w / 2, img_bgr.shape[1]), min(cy + h / 2, img_bgr.shape[0])])
            boxes = np.array(enlarged_boxes)
            # マスクも保持
            raw_sam3_masks = raw_sam3_output["masks"][keep].cpu().numpy()
        else:
            if 'detector' not in locals():
                detector = HumanDetector(name=args.detector_name, device=device, path=det_path)
                
            print(f"  -> Running {args.detector_name} inference (prompt: {args.text_prompt})...")
            boxes = detector.run_human_detection(img_bgr, bbox_thr=args.conf_threshold, nms_thr=args.nms_thr)
            raw_sam3_masks = None
            
    except Exception as e:
        print(f"❌ ERROR in Detection Initialization/Execution: {e}")
        boxes = np.array([[0, 0, img_bgr.shape[1], img_bgr.shape[0]]])
        raw_sam3_masks = None

    valid_masks = []
    print(f"  Total detected boxes: {len(boxes)}")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        area = int((x2 - x1) * (y2 - y1))
        
        # マスク生成: SAM3の場合は精密なマスクを使用、それ以外は矩形
        if raw_sam3_masks is not None and i < len(raw_sam3_masks):
            # 保存前に確実に 2次元 boolean に整形
            mask = raw_sam3_masks[i]
            if mask.ndim > 2:
                mask = np.squeeze(mask)
                if mask.ndim > 2: mask = mask[0]
            mask = mask.astype(bool)
        else:
            mask_uint8 = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask_uint8, (x1, y1), (x2, y2), 1, -1)
            mask = mask_uint8.astype(bool)

        if area > args.min_area:
            valid_masks.append({
                'id': int(i), 
                'segmentation': mask, 
                'area': area, 
                'score': 1.0, 
                'bbox': [float(x) for x in box] 
            })
        # ログ過多によるスタック防止のため、除外ログを抑制

    print(f"  Detected {len(valid_masks)} persons (after filtering).")
    
    for i, m in enumerate(valid_masks):
        dbg = img_bgr.copy()
        h, w = dbg.shape[:2]
        
        # 1. 人物の強調 (鮮やかなオレンジの縁取り + 非選択エリアの減光)
        # OpenCVのエラーとIndexErrorを避けるため、マスクを確実に 2次元・(H, W)・uint8 に変換
        mask_raw = m['segmentation']
        if mask_raw.ndim > 2:
            # (1, H, W) -> (H, W) 等、1の次元をすべて取り除く
            mask_raw = np.squeeze(mask_raw)
            if mask_raw.ndim > 2: # それでも多次元なら最初のチャンネルを選択
                mask_raw = mask_raw[0]

        mask_u8 = (mask_raw.astype(np.uint8)) * 255
        mask_bool = mask_raw.astype(bool)

        ctrs, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dbg, ctrs, -1, (0, 128, 255), 8) # 縁取り
        
        # 背景減光 (マスク外を 1/6 に)
        dbg[~mask_bool] = (dbg[~mask_bool] // 6).astype(np.uint8)
        
        # 2. 下部に「外枠」として巨大なIDバーを付け足す (センタリング)
        bar_h = int(h * 0.18)
        if bar_h < 80: bar_h = 80
        
        # キャンバスを拡張 (元の画像 + 黒帯)
        canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = dbg # 上部に元の画像(加工済み)を配置
        
        text = f"ID: {m['id']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = h / 350.0 
        if font_scale < 1.8: font_scale = 1.8
        thickness = int(font_scale * 2.5)
        
        (t_w, t_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        tx = max((w - t_w) // 2, 0)
        # 拡張した部分の中央に配置
        ty = h + (bar_h + t_h) // 2
        cv2.putText(canvas, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        cv2.imwrite(os.path.join(DEBUG_DIR, f"detected_person_{i}.jpg"), canvas)

    with open(os.path.join(OUTPUT_DIR, "detection_result.json"), "w") as f:
        json.dump([{'id': m['id'], 'area': m['area'], 'score': m['score'], 'bbox': m['bbox']} for m in valid_masks], f)
    
    # [Step 1 完了] メモリを徹底的に解放
    print("--- Cleaning up Step 1 memory ---")
    if 'detector' in locals():
        if hasattr(detector, 'detector'): del detector.detector
        if hasattr(detector, 'processor'): del detector.processor
        del detector
    if 'raw_sam3_output' in locals(): del raw_sam3_output
    if 'inference_state' in locals(): del inference_state
    if 'm_s3' in locals(): del m_s3
    if 'raw_sam3_masks' in locals(): del raw_sam3_masks
    clear_memory()
    print("--- Step 1 memory cleaned ---")

    if args.sam3_only:
        print(f"✅ SUCCESS. Detection complete in {time.time()-time_start:.2f}s.")
        sys.exit(0)

    # [Step 2] MoGe2: Depth
    depth_map = np.zeros(img_bgr.shape[:2], dtype=np.float32)
    if args.use_moge:
        print(f"--- [Step 2] MoGe2: Depth Estimation ---")
        import moge.model
        m_m = moge.model.import_model_class_by_version("v2").from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
        with torch.no_grad():
            img_rgb_t = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().to(device)/255.0
            inf_out = m_m.infer(img_rgb_t)
            depth_map = inf_out['depth'].cpu().numpy()
            depth_map = np.nan_to_num(depth_map, nan=0.0)
            
            # 深度マップを鮮明に可視化 (0を除いた有効範囲で正規化)
            valid_mask = (depth_map > 1e-3)
            if valid_mask.any():
                v_min = np.percentile(depth_map[valid_mask], 2)
                v_max = np.percentile(depth_map[valid_mask], 98)
                d_vis = np.clip((depth_map - v_min) / (v_max - v_min + 1e-8), 0, 1)
                # 有効領域以外（背景）は0にする
                d_vis[~valid_mask] = 0
            else:
                d_vis = depth_map
            
            d_vis = (d_vis * 255).astype(np.uint8)
            # 奥行きを直感的にするために色彩を調整
            cv2.imwrite(os.path.join(OUTPUT_DIR, "output_depth.jpg"), cv2.applyColorMap(d_vis, cv2.COLORMAP_JET))
            
        print("--- Cleaning up Step 2 memory ---")
        del m_m, img_rgb_t; clear_memory()
        print("--- Step 2 memory cleaned ---")

    # [Step 3] SAM 3DB: Estimation
    print(f"--- [Step 3] SAM 3DB: 3D Recovery (Mode: {args.inference_type}) ---")
    clear_memory()
    
    target_id_list = args.target_ids.split(",") if args.target_ids else []
    if target_id_list:
        to_p = [m for m in valid_masks if str(m['id']) in target_id_list]
    else:
        # オートモード（クイック復元等）: 検出された全員を処理対象にする
        to_p = valid_masks
    
    print(f"--- Targets to process: {[m['id'] for m in to_p]} (Total: {len(to_p)}) ---")
    
    if not to_p:
        print("⚠ No persons found or selected for 3D recovery.")
        sys.exit(0)

    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
    from sam_3d_body.metadata.mhr70 import pose_info
    
    print("--- Loading SAM 3D Body model... ---")
    model_3d, cfg_3d = load_sam_3d_body(SAM3DB_CKPT, device, MHR_MODEL_PT)
    est = SAM3DBodyEstimator(model_3d, cfg_3d)
    viz = SkeletonVisualizer(radius=4, line_width=2); viz.set_pose_meta(pose_info); v_img = img_bgr.copy()

    all_json_paths = []
    person_count_total = 0 # 複数人時のオフセット用
    for m in to_p:
        pid = m['id']; mask = m['segmentation']; tmp = os.path.join(OUTPUT_DIR, f"temp_{pid}.jpg")
        print(f"  -> Processing target ID {pid} (Index: {person_count_total})...")
        person_count_total += 1
        
        # マスクの形状補正ロジック (略)
        if mask.shape[:2] != img_bgr.shape[:2]:
            if mask.ndim > 2: mask = np.squeeze(mask)
            if mask.ndim != 2 or mask.shape != img_bgr.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        crp = img_bgr.copy()
        try:
            crp[~mask] = 0 
            cv2.imwrite(tmp, crp); del crp
        except Exception as e:
            print(f" ⚠ Warning: Mask shape mismatch for ID {pid}: {e}.")
            x1, y1, x2, y2 = map(int, m['bbox'])
            crp = crp[y1:y2, x1:x2]; cv2.imwrite(tmp, crp); del crp

        try:
            r = est.process_one_image(tmp, inference_type=args.inference_type)
            if not r:
                print(f"    ⚠ Warning: No prediction returned for ID {pid}")
                continue
            r = r[0] if isinstance(r, list) else r
            print(f"    ✅ Prediction success for ID {pid}")
            for k in r:
                if torch.is_tensor(r[k]): r[k] = r[k].cpu().numpy()
                if isinstance(r[k], np.ndarray): r[k] = np.nan_to_num(r[k], nan=0.0)
            
            if args.use_moge:
                # 3D空間上の位置を計算 (ピンホールカメラモデル)
                # ユーザー指定のFOVを使用
                fov_deg = args.fov
                h_img, w_img = img_bgr.shape[:2]
                f_pix = (max(w_img, h_img) / 2) / np.tan(np.deg2rad(fov_deg) / 2)
                cx_img, cy_img = w_img / 2, h_img / 2
                
                # 人物の2D中心 (bboxを使用)
                x1, y1, x2, y2 = m['bbox']
                px, py = (x1 + x2) / 2, (y1 + y2) / 2
                
                # 深度 (Z) の取得
                yc, _ = np.where(mask)
                if len(yc) > 0:
                    z_val = float(np.nanmedian(depth_map[mask & (np.arange(img_bgr.shape[0])[:, None] >= np.percentile(yc, 90))]))
                    # X, Y オフセットの計算
                    xoff = (px - cx_img) * z_val / f_pix
                    yoff = (py - cy_img) * z_val / f_pix
                    
                    # 3D座標の更新 (MHR形式: X, Y, Z)
                    for k in ['pred_keypoints_3d', 'pred_vertices']:
                        r[k][..., 0] += xoff
                        r[k][..., 1] += yoff
                        r[k][..., 2] += z_val
            else:
                # MoGe オフの場合: 
                # 複数人が重ならないよう、1.2m ずつ X 軸方向にずらして配置する
                offset_x = (person_count_total - 1) * 1.2
                if offset_x != 0:
                    print(f"    ℹ️ (MoGe Off) Applying horizontal offset: {offset_x:.2f}m")
                    for k in ['pred_keypoints_3d', 'pred_vertices']:
                        r[k][..., 0] += offset_x
            
            v_img = viz.draw_skeleton(v_img, np.hstack([r["pred_keypoints_2d"], np.ones((70,1))]))
            r['faces'] = est.faces; np.save(os.path.join(OUTPUT_DIR, f"output_{pid}.npy"), r)
            
            # [Step 4] Blender: FBX/BVH/OBJ Generation
            print(f"    --- [Step 4] Blender: FBX/BVH/OBJ Generation for ID {pid} ---")
            fbp = os.path.join(OUTPUT_DIR, f"output_{pid}.fbx"); tjp = os.path.join(OUTPUT_DIR, f"tjson_{pid}.json")
            all_json_paths.append(tjp)
            with open(tjp, 'w') as f: json.dump({"vertices": r["pred_vertices"].tolist(), "faces": est.faces.tolist(), "joints_mhr70": r["pred_keypoints_3d"].tolist()}, f)
            
            # FBX & BVH
            subprocess.run(["blender", "--background", "--python", os.path.join(BASE_DIR, "convert", "lib", "blender_humanoid_builder_v2.py"), "--", tjp, fbp], capture_output=False)
            if os.path.exists(fbp): 
                print(f"      ✅ FBX generated for ID {pid}")
                subprocess.run(["blender", "--background", "--python", os.path.join(BASE_DIR, "convert", "lib", "blender_bvh_crysta.py"), "--", fbp, os.path.join(OUTPUT_DIR, f"output_{pid}.bvh")], capture_output=False)
                if os.path.exists(os.path.join(OUTPUT_DIR, f"output_{pid}.bvh")):
                    print(f"      ✅ BVH generated for ID {pid}")
            else:
                print(f"      ⚠ FBX generation FAILED for ID {pid}")
            
            # OBJ (Static Mesh) - 以前の NameError: v を修正
            # f-string 内で backslash を使えない制約を回避するため事前にパスを加工
            obj_out = os.path.join(OUTPUT_DIR, f'output_{pid}.obj')
            obj_out_fixed = obj_out.replace('\\', '/')
            tjp_fixed = tjp.replace('\\', '/')
            # Blender 3.0.x (Colab) と 3.2+ (最新) の両対応
            blender_script_obj = f"""
import bpy,os,json,sys
try:
    import numpy as np
except ImportError:
    for p in ["/usr/local/lib/python3.10/dist-packages", "/usr/local/lib/python3.11/dist-packages", "/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages"]:
        if os.path.exists(p) and p not in sys.path: sys.path.append(p)
bpy.ops.wm.read_factory_settings(use_empty=True)
d=json.load(open('{tjp_fixed}'))
m=bpy.data.meshes.new('Mesh')
o=bpy.data.objects.new('Mesh',m)
bpy.context.collection.objects.link(o)
m.from_pydata([(v[0],v[2],-v[1]) for v in d['vertices']],[],d['faces'])
bpy.context.view_layer.objects.active=o
o.select_set(True)
try:
    bpy.ops.wm.obj_export(filepath='{obj_out_fixed}', export_selected_objects=True)
except (AttributeError, RuntimeError):
    bpy.ops.export_scene.obj(filepath='{obj_out_fixed}', use_selection=True)
"""
            subprocess.run(["blender", "--background", "--python-expr", blender_script_obj], capture_output=False)
            if os.path.exists(obj_out):
                print(f"      ✅ OBJ generated for ID {pid}")

        except Exception as e: print(f" Error {pid}: {e}")
        finally:
            if os.path.exists(tmp): os.remove(tmp)
            # 人物ごとの処理が終わるたびにメモリを解放 (Colabでの蓄積を防止)
            clear_memory()

    # [Step 5] Combined GLB for Preview
    if all_json_paths:
        print("--- [Step 5] Generating combined GLB for preview ---")
        comb_script = os.path.join(BASE_DIR, "convert", "lib", "blender_scene_combiner.py")
        ts = int(time.time())
        glb_out = os.path.join(OUTPUT_DIR, f"output_preview_combined_{ts}.glb")
        subprocess.run(["blender", "--background", "--python", comb_script, "--", glb_out] + all_json_paths, capture_output=False)
        if os.path.exists(glb_out):
            print("    ✅ Combined GLB generated.")
        else:
            print("    ⚠ Combined GLB generation FAILED.")
        # 不要な一時JSONを削除
        for p in all_json_paths: 
            if os.path.exists(p): os.remove(p)

    cv2.imwrite(VIS_OUTPUT, v_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "output_vis_skeleton.jpg"), v_img)
    print(f"✅ SUCCESS. Total time: {time.time()-time_start:.2f}s")
