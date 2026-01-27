#!/bin/bash
# setup_colab.sh - Google Colab 用セットアップスクリプト

echo "🚀 SAM 3D Pose Analyzer の環境を構築中..."

# 1. システムライブラリのインストール
apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    libosmesa6 libosmesa6-dev libglu1-mesa freeglut3-dev \
    blender

# 2. Python 依存関係のインストール
# utils3d や triton の競合回避のため、先に個別にインストールします
pip install git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183 --no-deps
pip install triton
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gdown "huggingface_hub<1.0"

# 3. 外部リポジトリのセットアップ (Git管理から外したコードを再取得)
mkdir -p repos
pushd repos
[ ! -d "sam-3d-body" ] && git clone https://github.com/facebookresearch/sam-3d-body.git
[ ! -d "sam3" ] && git clone https://github.com/facebookresearch/sam3.git
[ ! -d "MoGe" ] && git clone https://github.com/microsoft/MoGe.git
popd

# 4. モデルのダウンロード (Python スクリプトで確実に実行)
mkdir -p weights/body/assets
echo "📦 モデルチェックポイントを準備中..."

cat <<EOF > download_models.py
import os
from huggingface_hub import hf_hub_download

def download():
    # SAM 3D Body models
    print("Downloading SAM 3D Body models...")
    hf_hub_download(repo_id="facebook/sam-3d-body-dinov3", filename="model.ckpt", local_dir="weights/body")
    hf_hub_download(repo_id="facebook/sam-3d-body-dinov3", filename="assets/mhr_model.pt", local_dir="weights/body")
    
    # SAM 3 model
    if not os.path.exists("weights/body/sam3.pt"):
        print("Downloading SAM 3 model...")
        path = hf_hub_download(repo_id="facebook/sam3", filename="model.pt", local_dir="weights/body")
        os.rename(path, "weights/body/sam3.pt")

if __name__ == "__main__":
    download()
EOF

python3 download_models.py
rm download_models.py

echo "✅ セットアップ完了！"
