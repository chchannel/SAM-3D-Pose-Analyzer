import os
import sys
import subprocess
import tempfile
import shutil
import glob
import json
from datetime import datetime
import gradio as gr

# パス設定
base_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(base_dir, "outputs")
debug_dir = os.path.join(outputs_dir, "debug_masks")
settings_path = os.path.join(base_dir, "settings.json")
os.makedirs(outputs_dir, exist_ok=True)

# 🚀 実行中プロセスの管理
running_processes = []

def kill_running_processes():
    global running_processes
    for p in running_processes:
        try:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=1)
                print(f"Process {p.pid} terminated.")
        except:
            try: p.kill()
            except: pass
    running_processes = []
    return "⏹️ 処理を中断しました。"

def load_settings():
    default_settings = {
        "detector_name": "sam3", "text_prompt": "person", "conf_threshold": 0.5, "min_area": 1000,
        "inference_type": "full (body+hand)",
        "use_moge": True,
        "clear_mem": True,
        "fov": 70.0,
        "box_scale": 1.2,
        "nms_thr": 0.3
    }
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                loaded = json.load(f); default_settings.update(loaded)
        except: pass
    return default_settings

def save_settings_fn(detector, text_prompt, conf_threshold, min_area, inference_type, use_moge, clear_mem, fov, box_scale, nms_thr):
    settings = {
        "detector_name": detector, "text_prompt": text_prompt, "conf_threshold": conf_threshold, "min_area": min_area,
        "inference_type": inference_type, "use_moge": use_moge, "clear_mem": clear_mem,
        "fov": fov, "box_scale": box_scale, "nms_thr": nms_thr
    }
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)
    return "✅ 設定をデフォルトとして保存しました"

def run_worker_cmd_yield(cmd, desc):
    global running_processes
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    running_processes.append(process)
    
    full_log = f"--- [START] {desc} ---\n"
    yield full_log
    
    for line in iter(process.stdout.readline, ""):
        full_log += line
        print(line, end="")
        yield full_log
    
    process.wait()
    if process in running_processes: running_processes.remove(process)
    if process.returncode != 0:
        yield full_log + f"\n❌ ERROR: 終了コード {process.returncode}\n"
    else:
        yield full_log + f"\n✅ SUCCESS: 完了\n"

def create_app():
    defaults = load_settings()
    worker_script = os.path.join(base_dir, "predict_worker.py")

    with gr.Blocks(title="SAM 3D : Pose Analyzer") as app:
        gr.Markdown("# 🧍 SAM 3D : Pose Analyzer")
        
        with gr.Accordion("📖 はじめにお読みください (ステップ・バイ・ステップ)", open=True):
            gr.Markdown("""
#### 🛠️ 全体の流れ
1.  **[1. Detection] タブ**で画像をスキャンし、3D化したい人物を探します。
2.  スキャン結果から人物を選び（IDにチェック）、**[2. 3D Recovery] タブ**に移動します。
3.  **[2. 3D Recovery] タブ**で復元ボタンを押し、3Dデータを生成・確認します。
""")

        session_id = gr.State("")

        with gr.Tabs() as tabs:
            # --- Tab 1: 検出 ---
            with gr.TabItem("1. Detection & Select (人物のスキャン)", id=0):
                with gr.Row():
                    with gr.Column(scale=1): # 左重心
                        input_img = gr.Image(label="入力画像", type="filepath", height=280)
                        
                        gr.Markdown("### 🎯 生成対象の選択")
                        with gr.Group():
                            target_id_checks = gr.CheckboxGroup(label="対象 ID (検出後にチェック)", choices=[], value=[])
                            with gr.Row():
                                select_all_btn = gr.Button("全て選択", size="sm")
                                deselect_all_btn = gr.Button("全て解除", size="sm")

                        gr.Markdown("### 🔍 検出設定")
                        with gr.Group():
                            detector_sel = gr.Dropdown(
                                ["sam3", "vitdet"], 
                                value=defaults["detector_name"], 
                                label="検出モデル",
                                info="人物を切り出すAIを選びます。sam3は服や小物の精度が高いですが少し時間がかかります。"
                            )
                            text_prompt = gr.Textbox(
                                value=defaults["text_prompt"], 
                                label="検索ターゲット",
                                info="検出したいものを言葉で指定します。通常は 'person' でOKです。"
                            )
                            conf_threshold = gr.Slider(
                                0.1, 1.0, 
                                value=defaults["conf_threshold"], 
                                label="検出感度 (Confidence)",
                                info="値を下げると検出しやすくなりますが、人間以外を誤検出する可能性も増えます。"
                            )
                            min_area = gr.Slider(
                                500, 50000, 
                                value=defaults["min_area"], 
                                step=500, 
                                label="除外サイズ (Min Area)",
                                info="この数値より小さい（遠くにいる）人物は無視します。"
                            )
                            with gr.Accordion("🛠️ 検出アドバンス設定", open=False):
                                box_scale = gr.Slider(
                                    1.0, 2.0, 
                                    value=defaults["box_scale"], 
                                    step=0.1,
                                    label="ボックスの余白 (Box Scale)",
                                    info="人物をどれくらい広めに切り出すか。姿勢推定の精度に影響します。"
                                )
                                nms_thr = gr.Slider(
                                    0.1, 1.0, 
                                    value=defaults["nms_thr"], 
                                    label="重複除去 (NMS Threshold)",
                                    info="値が小さいほど、重なり合った人物の重複検出を厳しく削除します。"
                                )
                        
                        with gr.Row():
                            det_btn = gr.Button("🔍 検出開始", variant="primary", scale=2)
                            cancel_det_btn = gr.Button("⏹️ 停止", variant="stop", scale=1)
                        save_settings_btn1 = gr.Button("💾 デフォルトとして保存", size="sm")
                        
                    with gr.Column(scale=3):
                        det_preview = gr.Gallery(label="ID付きプレビュー", columns=3, height="auto")
                        gr.Markdown("""
### ⏭️ 次のステップ (重要)
1. 上の画像で、推論したい人物の **ID (番号)** を探します。
2. 左側の **[対象 ID]** リストで、その番号にチェックを入れます。
3. 画面最上部のオレンジ色のバーまたはタブをクリックし、**『2. 3D Recovery』へ移動**してください。
""")
                        det_status_msg = gr.Markdown("")
                        det_results_json = gr.JSON(label="検出詳細", visible=False)

            # --- Tab 2: 3D復元・出力 ---
            with gr.TabItem("2. 3D Recovery (モデル生成と確認)", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 推論設定")
                        inf_type = gr.Dropdown(
                            ["full (body+hand)", "body", "hand"], 
                            value=defaults["inference_type"], 
                            label="推論モード",
                            info="bodyは全身のみ、fullは指先まで細かく復元を試みます。"
                        )
                        use_moge = gr.Checkbox(
                            value=defaults["use_moge"], 
                            label="空間配置 (MoGe2) 有効",
                            info="写真を解析して、3D空間上の正しい位置に人物を立たせます。"
                        )
                        clear_mem = gr.Checkbox(
                            value=defaults["clear_mem"], 
                            label="VRAMメモリ解放",
                            info="完了ごとにメモリを掃除します。GPUメモリが少ない(8GB以下)場合はON推奨です。"
                        )
                        
                        gr.Markdown("### 📏 空間配置設定")
                        with gr.Group():
                            fov_slider = gr.Slider(
                                30, 120, 
                                value=defaults["fov"], 
                                step=1,
                                label="カメラ画角 (FOV)",
                                info="広角レンズ(iPhone等)なら70~80、標準なら50前後に調整してください。"
                            )
                        
                        with gr.Row():
                            run_3d_btn = gr.Button("🚀 3D復元開始", variant="primary", scale=2)
                            cancel_3d_btn = gr.Button("⏹️ 停止", variant="stop", scale=1)
                        
                        save_settings_btn2 = gr.Button("💾 設定保存", size="sm")

                        gr.HTML("<hr>")
                        gr.Markdown("### 📂 生成ファイル")
                        output_bvh = gr.File(label="BVH (Motion)", file_count="multiple")
                        output_fbx = gr.File(label="FBX (Mesh)", file_count="multiple")
                        output_obj = gr.File(label="OBJ (Static Mesh)", file_count="multiple")
                        open_folder_btn = gr.Button("📂 フォルダを開く", size="sm")

                        gr.HTML("<hr>")
                        gr.Markdown("### 📜 実行ログ")
                        log_output = gr.Textbox(label="", lines=12, max_lines=20, interactive=False)

                    with gr.Column(scale=3):
                        gr.Markdown("### 🖼️ プレビュー (v0.5 暫定版)")
                        with gr.Group():
                            with gr.Row():
                                vis_skeleton = gr.Image(label="スケルトン (Pose/Exact)")
                                vis_moge = gr.Image(label="深度マップ (MoGe/Exact)")
                            interactive_3d = gr.Model3D(label="3D プレビュー (回転・拡大可能)", height=500)
                        
                        with gr.Group():
                            gr.Markdown("""> [!IMPORTANT]
> **💡 画面が真っ白で 3D が見えない場合**
> ブラウザの **ハードウェアアクセラレーション** がオフになっている可能性があります。
> 設定から「グラフィックアクセラレーションを使用する」をオンにして再起動してください。""")
                            gr.Markdown("""> [!NOTE]
> **3Dプレビューについて**: 上記の 3D プレビューはマウスで自由に**回転・ズーム**が可能です。
> 背景画像との重畳プレビュー（Mesh Overlay）がズレる場合は、こちらで復元された 3D 形状を確認してください。""")
                        
                        with gr.Accordion("❓ 設定項目の詳細説明", open=False):
                            gr.Markdown("""
#### 🎯 検出設定 (Detection & Select)
- **検出モデル**: 
    - `sam3`: 精密。服や持ち物を含めた切り出しが最も綺麗ですが、グラフィックボードの負荷が高いです。
    - `vitdet`: 標準的。高速ですが、境界が sam3 より少しラフになることがあります。
- **検索ターゲット**: AIに何を探させるか。基本は `person` です。
- **検出感度 (Confidence)**: 1に近いほど「確実なもの」だけを検出します。0に近いほど「人間に見えるもの」を広く拾います。
- **除外サイズ (Min Area)**: 小さなゴミや、背景の豆粒のような人物を無視するために使います。

#### ⚙️ 推論設定 (3D Recovery)
- **推論モード**:
    - `body`: 体の主要な関節のみを復元します。
    - `full`: 指先の動きまで復元を試みます。処理時間は少し増えます。
- **空間配置 (MoGe2)**: 写真の「奥行き」をAIが推測し、人物を正しい地面・距離に立たせます。
- **VRAMメモリ解放**: 処理が終わるたびに掃除をします。VRAMが8GBのカード（3060Ti等）では常にONを推奨します。
""")
                        
                        with gr.Accordion("📜 Credits & License", open=False):
                            gr.Markdown("""
This tool integrates the following research works:
- **SAM 3D Body**: [Meta Research] (SAM License)
- **MoGe**: [Microsoft Research] (MIT License)
- **Detectron2**: [Meta AI] (Apache 2.0)
- **Gradio Wrapper**: Copyright (c) 2026 Author (Proprietary License)
    - ツール自体の無断商用利用・再配布は禁止します。
    - **本ツールで生成したデータ（3Dモデル等）は商用利用可能です。**
""")
                        
                        status_msg = gr.Markdown("")

        # --- Logic ---
        def on_detect(image, detector, text, conf, area, b_scale, nms):
            if not image: yield [], {}, "", gr.update(choices=[], value=[]), "画像なし", ""
            cmd = [sys.executable, worker_script, image, "--detector_name", detector, "--text_prompt", text, "--conf_threshold", str(conf), "--min_area", str(int(area)), "--box_scale", str(b_scale), "--nms_thr", str(nms), "--sam3_only"]
            log_c = ""
            success = False
            for log_c in run_worker_cmd_yield(cmd, "人物検出"):
                yield [], {}, "", gr.update(), "🚀 実行中...", log_c
                if "✅ SUCCESS" in log_c: success = True
            
            if not success:
                yield [], {}, "", gr.update(choices=[], value=[]), "❌ 失敗", log_c
                return

            previews = sorted(glob.glob(os.path.join(debug_dir, "*.jpg")))
            det_data = []
            if os.path.exists(os.path.join(outputs_dir, "detection_result.json")):
                with open(os.path.join(outputs_dir, "detection_result.json"), "r") as f:
                    det_data = json.load(f)
            choices = [str(d['id']) for d in det_data]
            yield previews, det_data, datetime.now().strftime("%H%M%S"), gr.update(choices=choices, value=choices), "✅ 完了", log_c
 
        det_job = det_btn.click(on_detect, [input_img, detector_sel, text_prompt, conf_threshold, min_area, box_scale, nms_thr], [det_preview, det_results_json, session_id, target_id_checks, det_status_msg, log_output])
        cancel_det_btn.click(kill_running_processes, None, [log_output], cancels=[det_job])

        select_all_btn.click(lambda x: [str(d['id']) for d in x] if x else [], [det_results_json], [target_id_checks])
        deselect_all_btn.click(lambda: [], None, [target_id_checks])

        def on_3d_recovery(image, detector, text, conf, area, b_scale, nms, targets, inf_mode, moge_active, clear, fov):
            if not image or not targets: yield None, None, None, [], [], [], "対象選択なし", ""
            real_inf_mode = "full" if "full" in inf_mode else inf_mode
            cmd = [sys.executable, worker_script, image, "--detector_name", detector, "--text_prompt", text, "--conf_threshold", str(conf), "--min_area", str(int(area)), "--box_scale", str(b_scale), "--nms_thr", str(nms), "--inference_type", real_inf_mode, "--fov", str(fov)]
            if moge_active: cmd.append("--use_moge")
            if clear: cmd.append("--clear_mem")
            cmd.extend(["--target_ids", ",".join(targets)])
            log_c = ""
            success = False
            for log_c in run_worker_cmd_yield(cmd, "3D復元処理"):
                yield None, None, None, [], [], [], "🚀 実行中...", log_c
                if "✅ SUCCESS" in log_c: success = True
            
            if not success:
                yield None, None, None, [], [], [], "❌ 失敗", log_c
                return

            v_skel = os.path.join(outputs_dir, "output_vis_skeleton.jpg")
            v_moge = os.path.join(outputs_dir, "output_depth.jpg")
            bvh = sorted(glob.glob(os.path.join(outputs_dir, "output_*.bvh")))
            fbx = sorted(glob.glob(os.path.join(outputs_dir, "output_*.fbx")))
            obj = sorted(glob.glob(os.path.join(outputs_dir, "output_*.obj")))
            preview_glb = os.path.join(outputs_dir, "output_preview_combined.glb")
            
            # プレビュー用の統合GLBを表示
            target_glb = preview_glb if os.path.exists(preview_glb) else None
            
            if not fbx and not bvh:
                yield None, None, None, [], [], [], "⚠ 完了（ファイルが生成されませんでした）", log_c
            else:
                yield v_skel if os.path.exists(v_skel) else None, v_moge if os.path.exists(v_moge) else None, target_glb, bvh, fbx, obj, "✅ 完了", log_c

        rec_job = run_3d_btn.click(on_3d_recovery, [input_img, detector_sel, text_prompt, conf_threshold, min_area, box_scale, nms_thr, target_id_checks, inf_type, use_moge, clear_mem, fov_slider], [vis_skeleton, vis_moge, interactive_3d, output_bvh, output_fbx, output_obj, status_msg, log_output])
        cancel_3d_btn.click(kill_running_processes, None, [log_output], cancels=[rec_job])
 
        for b in [save_settings_btn1, save_settings_btn2]:
            b.click(save_settings_fn, [detector_sel, text_prompt, conf_threshold, min_area, inf_type, use_moge, clear_mem, fov_slider, box_scale, nms_thr], [status_msg])
        
        open_folder_btn.click(lambda: subprocess.run(["explorer.exe", "."], cwd=outputs_dir), None, None)

    return app

if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Enable Gradio public link")
    args = parser.parse_args()

    # Hugging Face Spaces や Docker 環境用の設定
    server_port = int(os.environ.get("PORT", 7860))
    
    print("\n" + "="*60)
    print("🚀 SAM 3D Pose Analyzer を起動しています...")
    print("Google Colab をご利用の場合、以下の 'public URL' をクリックしてください。")
    print("※ 'local URL' は Colab では接続できません。")
    print("="*60 + "\n")

    create_app().launch(
        server_name="0.0.0.0", 
        server_port=server_port, 
        share=args.share,
        allowed_paths=[outputs_dir]
    )
