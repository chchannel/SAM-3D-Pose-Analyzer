# SAM 3D Pose Analyzer

[![v0.5.1](https://img.shields.io/badge/version-v0.5.1-blue.svg)](https://github.com/chchannel/SAM-3D-Pose-Analyzer)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-orange.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

SAM 3D Pose Analyzer は、Meta Research の **SAM 3D Body** をベースに、単一画像から 3D 人体ポーズおよびメッシュを抽出するための統合ソリューションです。

本ツールは、画像から人物を検出し、各種 DCC ツール（Blender, Clip Studio Paint, Unity 等）で即座に利用可能なアセットを出力します。

## 🎯 主なユースケース

- **Blender / Maya**: ボーンおよびスキニング済みメッシュの抽出 (FBX 形式)
- **Clip Studio Paint**: 3D デッサン人形用ポーズデータ (BVH 形式)
    - ※出力された BVH をキャンバス上の 3D デッサン人形にドラッグ＆ドロップすることで即座にポーズが適用されます。
- **Unity / Unreal Engine / その他**: 3D リファレンス用アセット (OBJ, GLB 形式)

## ✨ サポートしている拡張子

- **FBX**: アニメーション用ボーン + スキニング済みメッシュ
- **BVH**: ポーズデータ
- **OBJ**: 静止メッシュデータ
- **GLB**: Web/AR 用バイナリ形式

## 🚀 実行方法 (Quick Start)

### 1. Google Colab
- [**SAM 3D Pose Analyzer on Colab**](https://colab.research.google.com/github/chchannel/SAM-3D-Pose-Analyzer/blob/main/sam_3d_pose_analyzer_colab.ipynb)
    - ※ノートブックを開き、各セルを順に実行してください

### 2. ローカル環境 (Local Installation)
WSL2 または Linux 環境での動作を想定しています。

リポジトリを軽量化しているため、初回実行前に外部リポジトリとモデルの取得が必要です。
```bash
# リポジトリの取得
git clone https://github.com/chchannel/SAM-3D-Pose-Analyzer.git
cd SAM-3D-Pose-Analyzer

# 依存ライブラリのインストール
pip install -r requirements.txt

# 外部リポジトリとモデルのセットアップ（初回のみ）
# 注意: Blender がインストールされている必要があります
bash setup_colab.sh 

# アプリの起動
python app/main.py
```
> [!NOTE]
> `setup_colab.sh` は Colab 用ですが、WSL2/Linux 環境でも外部リポジトリの取得やモデルのダウンロードに利用可能です。

## 📜 ライセンス (Licensing)

- **生成データ (Output Assets)**: 商用・非商用を問わず、**自由にご利用いただけます。**
- **ソースコード (This Repository)**: 非商用利用に限定され、無断再配布は禁止されています。
- **技術基盤**: 以下の各公式リポジトリのライセンス条件を継承します。
    - [SAM 3D Body (Meta)](https://github.com/facebookresearch/sam-3d-body)
    - [SAM 3 (Meta)](https://github.com/facebookresearch/sam3)
    - [MoGe (Microsoft)](https://github.com/microsoft/MoGe)
    - [Detectron2 (Meta)](https://github.com/facebookresearch/detectron2)

## 🤝 謝辞 (Acknowledgments / Attribution)

本ツールの開発にあたり、以下のプロジェクトのコードを利用・参考にさせていただいています。

- **BVH I/O Logic**:
    - [smpl2bvh](https://github.com/KosukeFukazawa/smpl2bvh) (MIT License) - by Kosuke Fukazawa
    - [Motion-Matching](https://github.com/orangeduck/Motion-Matching) (MIT License) - by Daniel Holden
- **Blender 3D Export Idea**:
    - [note: SAM 3D BodyのポーズをBlenderで再現する](https://note.com/tori29umai/n/n5550b2b5ec26) - by とり

---
*Developed by Antigravity (AI Assistant) & USER.*
