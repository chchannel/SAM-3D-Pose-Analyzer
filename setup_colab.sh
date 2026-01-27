#!/bin/bash
# setup_colab.sh - Google Colab 用セットアップスクリプト

echo "🚀 SAM 3D Pose Analyzer の環境を構築中..."

# 1. システムライブラリのインストール
apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    libosmesa6 libosmesa6-dev libglu1-mesa freeglut3-dev \
    blender

# 2. Python 依存関係のインストール (requirements.txt を使用)
# torch は Colab に既にある場合はスキップされるが、念のため index 指定
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gdown

# 3. 外部リポジトリのセットアップ (Git管理から外したコードを再取得)
mkdir -p repos
pushd repos
[ ! -d "sam-3d-body" ] && git clone https://github.com/facebookresearch/sam-3d-body.git
[ ! -d "sam3" ] && git clone https://github.com/facebookresearch/sam3.git
[ ! -d "MoGe" ] && git clone https://github.com/microsoft/MoGe.git
popd

# 4. モデルのダウンロード (Hugging Face から取得)
mkdir -p weights/body/assets
echo "📦 モデルチェックポイントを準備中..."

# huggingface-cli を使用して facebook/sam-3d-body-dinov3 から取得
pip install -U "huggingface_hub[cli]"
huggingface-cli download facebook/sam-3d-body-dinov3 model.ckpt --local-dir weights/body --local-dir-use-symlinks False
huggingface-cli download facebook/sam-3d-body-dinov3 assets/mhr_model.pt --local-dir weights/body --local-dir-use-symlinks False

# SAM3 のデフォルトチェックポイント (HumanDetector 用)
if [ ! -f "weights/body/sam3.pt" ]; then
    echo "📦 SAM3 モデルをダウンロード中..."
    huggingface-cli download facebook/sam3 model.pt --local-dir weights/body --local-dir-use-symlinks False
    mv weights/body/model.pt weights/body/sam3.pt
fi

echo "✅ セットアップ完了！"
