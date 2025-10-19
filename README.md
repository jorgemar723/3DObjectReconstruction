# CS4337 — 3D Object Reconstruction from Images

## Overview
This project implements a **multi-view 3D reconstruction pipeline** using the [ShapeNetCore](https://shapenet.org) dataset.  
The pipeline demonstrates: **PyTorch3D rendering → SIFT feature detection → feature matching → camera pose estimation → triangulation → sparse 3D point cloud reconstruction**.

**Course**: CS 4337 — Introduction to Computer Vision (Fall 2025)  
**Instructor**: Dr. Bhandari  
**Team Members**: Jorge Martinez-Lopez, Kristian Parra  

---

## Project Goals
- Render multi-view images from ShapeNet 3D models using [PyTorch3D](https://pytorch3d.org/)
- Detect keypoints using SIFT feature detection (OpenCV)
- Match features between views using Lowe's ratio test
- Estimate camera poses via essential matrix and RANSAC
- Triangulate matched points to reconstruct sparse 3D point clouds
- Visualize results in MeshLab and matplotlib

---

## Repository Structure
```
3DObjectReconstruction/
│
├── notebooks/
│   └── render_shapenet_views.ipynb    # Main pipeline (rendering + reconstruction)
│
├── outputs/
│   ├── chair_views/                   # 8 rendered chair views
│   ├── car_views/                     # 8 rendered car views
│   └── chair_reconstruction.ply       # Sparse 3D point cloud
│
├── README.md
└── .gitignore
```

**Note**: ShapeNet dataset files are **NOT** included in this repo due to size (chairs: 1.83GB, cars: 5.30GB).  
Download from [HuggingFace ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) if needed.
- Note: Permission is required to view ShapeNet datasets.

---

## Setup Instructions

### Prerequisites
- **Google Colab Pro** account (recommended for GPU access)
- **HuggingFace** account (for ShapeNet dataset access)
- **MeshLab** (optional, for local PLY visualization)

### Running the Project

#### 1. Open the notebook in Google Colab
- Go to: [Google Colab](https://colab.research.google.com/)
- Click **File → Open notebook → GitHub**
- Enter: `jorgemar723/3DObjectReconstruction`
- Select: `notebooks/render_shapenet_views.ipynb`

#### 2. Set up Colab runtime
- **Runtime → Change runtime type → T4 GPU**
- This ensures PyTorch3D runs efficiently

#### 3. Install dependencies
The first cell in the notebook installs everything:
```python
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install fvcore iopath
!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py312_cu121_pyt280/download.html
!pip install opencv-python matplotlib numpy trimesh
```

#### 4. Mount Google Drive (optional but recommended)
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### 5. Run the notebook
- **Runtime → Run all**
- The pipeline will:
  1. Download ShapeNet chair/car models (~7GB total)
  2. Render 8 views of each object
  3. Detect SIFT keypoints
  4. Match features between views
  5. Estimate camera poses
  6. Triangulate to 3D point cloud
  7. Export `.ply` file for visualization

---

## Pipeline Overview

### Step 1: Data Loading
- Download ShapeNet chairs (`03001627.zip`) and cars (`02958343.zip`)
- Extract a single model for reconstruction

### Step 2: Multi-View Rendering
- Use PyTorch3D to render 8 views at different azimuths (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
- Camera distance: 1.0-1.2 units
- Resolution: 512x512 pixels

### Step 3: Feature Detection
- SIFT keypoint detection on each view
- Extract descriptors for matching

### Step 4: Feature Matching
- Brute-force matcher with Lowe's ratio test (threshold: 0.75)
- Filter matches between view pairs

### Step 5: Pose Estimation
- Compute essential matrix using RANSAC
- Recover rotation (R) and translation (t) between cameras

### Step 6: 3D Reconstruction
- Triangulate matched points using camera projection matrices
- Generate sparse 3D point cloud

### Step 7: Visualization
- Plot 3D point cloud with matplotlib
- Export to `.ply` format for MeshLab

---

## Results

### Chair Reconstruction
- **Input**: 8 rendered views of ShapeNet chair model
- **Output**: 13-point sparse 3D reconstruction
- **File**: `outputs/chair_reconstruction.ply`

### Car Reconstruction
- **Input**: 8 rendered views of ShapeNet car model (52,081 vertices)
- **Output**: Colored multi-view renders

---

## Viewing Results

### In Colab (automatic)
- Point clouds display in matplotlib 3D plots
- Rendered views show in subplot grids

### In MeshLab (local)
1. Download `chair_reconstruction.ply` from Colab
2. Open MeshLab
3. **File → Import Mesh → Select .ply file**
4. Adjust point size: **Render → Show Points (increase point size)**

---

## Technical Details

### Key Libraries
- **PyTorch3D**: 3D rendering and mesh operations
- **OpenCV**: SIFT feature detection, matching, pose estimation
- **NumPy**: Matrix operations and numerical computing
- **Matplotlib**: Visualization
- **Trimesh**: PLY file export

### Algorithms Implemented
- **SIFT** (Scale-Invariant Feature Transform) for keypoint detection
- **Lowe's ratio test** for robust feature matching
- **RANSAC** (Random Sample Consensus) for outlier rejection
- **Essential matrix decomposition** for camera pose recovery
- **Triangulation** for 3D point reconstruction

### Camera Model
- Pinned-hole camera with known intrinsics
- Focal length estimated from image width
- Principal point at image center

---

## Known Limitations
- **Sparse reconstruction**: Only matched keypoints are reconstructed (13 points for chair)
- **No texture mapping**: Grayscale/monochrome renders only
- **Limited views**: 8 views may not be enough for complex objects
- **ShapeNet only**: Pipeline tested only on synthetic data

### Future Improvements
- Dense reconstruction using multi-view stereo
- Texture mapping from original ShapeNet materials
- Real-world image processing (photos instead of renders)
- Interactive web viewer with Three.js

---

## Deliverables
- Jupyter notebook with complete pipeline
- 8 rendered views per object (chair + car)
- Feature matching visualizations
- Sparse 3D point cloud (`.ply` format)
- Final project report (not finished yet)
- Video demonstration (not finished yet)

---

## References
- [ShapeNet Dataset](https://shapenet.org)
- [PyTorch3D Documentation](https://pytorch3d.org/)
- [OpenCV Feature Detection Tutorial](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
- [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/) (Hartley & Zisserman)

---

## License
This project is for educational purposes as part of CS 4337 at Texas State University.
