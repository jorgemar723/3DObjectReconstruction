# CS4337 — 3D Object Reconstruction from Images

## Overview
This project implements a **3D object reconstruction pipeline** using images rendered from the [ShapeNetCore](https://shapenet.org) dataset.  
The pipeline demonstrates how to go from **2D images → feature matching → camera pose estimation → sparse 3D reconstruction → point cloud visualization**.

**Course**: CS 4337 — Introduction to Computer Vision (Fall 2025)  
**Team Members**: Jorge Martinez-Lopez, Kristian Parra.  

---

## Goals
- Render multi-view images from ShapeNet meshes with [PyTorch3D](https://pytorch3d.org/).  
- Detect and match keypoints between views using OpenCV (SIFT, ORB).  
- Estimate relative camera poses (essential matrix, recoverPose).  
- Triangulate matched points into a sparse 3D point cloud.  
- Visualize and refine results in MeshLab.  

---

## Repository Structure

```
3DObjectReconstruction/
│
├── data/
│ ├── shapenet/ # ShapeNet meshes (not tracked in git)
│ └── rendered/ # rendered images + cameras.json (not tracked in git)
│
├── notebooks/
│ ├── render_shapenet_views.ipynb # Colab rendering (PyTorch3D)
│ ├── keypoints.ipynb # ORB/SIFT keypoints on single image
│ └── matching.ipynb # feature matching between two views
│
├── src/
│ └── cameras.py # helper functions to load cameras.json
│
├── outputs/ # saved keypoint visualizations, match previews, .ply point clouds
│
├── README.md
├── requirements.txt
└── .gitignore

```


---

## Setup

### Local Environment (for OpenCV notebooks)
```
conda create -n CS4337 python=3.10 -y
conda activate CS4337
conda install -c conda-forge opencv numpy matplotlib scikit-image trimesh -y
```

## Google Colab (for PyTorch3D rendering)

Open notebooks/render_shapenet_views.ipynb in Colab.

Set runtime → GPU.

Install dependencies in the first cell:

```
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
!pip install opencv-python matplotlib numpy trimesh
```

Mount Google Drive if storing ShapeNet there:

```
from google.colab import drive
drive.mount('/content/drive')
```

### Deliverables

Keypoint visualizations (outputs/orb_keypoints.jpg, outputs/sift_keypoints.jpg).

Feature match visualization (outputs/matches_preview.jpg).

Sparse point cloud (outputs/reconstruction.ply).

Final demo video + report (submitted via Canvas).

### Notes

ShapeNet data is not included in this repo. Store meshes under data/shapenet/ and keep that folder gitignored.

For reproducibility, always save camera intrinsics/extrinsics to cameras.json alongside rendered images.

MeshLab is used for visualization of .ply point clouds.
