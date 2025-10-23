# CS4337 — 3D Object Reconstruction from Images

## Overview
This project implements a **multi-view 3D reconstruction pipeline** using both synthetic data from [ShapeNetCore](https://shapenet.org) and real-world photographs.  
The pipeline demonstrates: **PyTorch3D rendering → SIFT feature detection → feature matching → camera pose estimation → triangulation → sparse 3D point cloud reconstruction → interactive web visualization**.

**Course**: CS 4337 — Introduction to Computer Vision (Fall 2025)  
**Instructor**: Dr. Bhandari  
**Team Members**: Jorge Martinez-Lopez, Kristian Parra  

---

## Project Goals
- Render multi-view images from ShapeNet 3D models using [PyTorch3D](https://pytorch3d.org/)
- Apply the same pipeline to real-world photographs
- Detect keypoints using SIFT feature detection (OpenCV)
- Match features between views using Lowe's ratio test
- Estimate camera poses via essential matrix and RANSAC
- Triangulate matched points to reconstruct sparse 3D point clouds
- Create interactive web viewer for 3D visualizations
- Visualize results in MeshLab and matplotlib

---

## Repository Structure
```
3DObjectReconstruction/
│
├── notebooks/
│   ├── render_shapenet_views.ipynb        # Base pipeline (ShapeNet rendering + reconstruction)
│   ├── real_photos_reconstruction.ipynb   # Real-world photo reconstruction
│   └── web_viewer_setup.ipynb             # Three.js web viewer generation
│
├── outputs/
│   ├── chair_views/                       # 8 rendered ShapeNet chair views
│   ├── car_views/                         # 8 rendered ShapeNet car views
│   ├── real_object_views/                 # Real-world object photos
│   ├── chair_reconstruction.ply           # Sparse 3D point cloud (chair)
│   └── real_object_reconstruction.ply     # Sparse 3D point cloud (real object)
│
├── web_viewer/                            # Interactive 3D viewer
│   ├── index.html                         # Three.js viewer interface
│   └── reconstruction.json                # Point cloud data for web rendering
│
├── README.md
└── .gitignore
```

**Note**: ShapeNet dataset files are **NOT** included in this repo due to size (chairs: 1.83GB, cars: 5.30GB).
**Second Note** ShapeNet may require permission to view.
Download from [HuggingFace ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) if reproducing ShapeNet experiments.

---

## Setup Instructions

### Prerequisites
- **Google Colab** account (free tier sufficient for real photos, Pro recommended for ShapeNet)
- **HuggingFace** account (only needed for ShapeNet dataset access)
- **MeshLab** (optional, for local PLY visualization)
- **Phone camera** (for real-world photo reconstruction)

### Running the Project

#### Option 1: ShapeNet Synthetic Data (Complete Pipeline)

**1. Open the base notebook in Google Colab**
- Go to: [Google Colab](https://colab.research.google.com/)
- Click **File → Open notebook → GitHub**
- Enter: `jorgemar723/3DObjectReconstruction`
- Select: `notebooks/render_shapenet_views.ipynb`

**2. Set up Colab runtime**
- **Runtime → Change runtime type → T4 GPU**
- This ensures PyTorch3D runs efficiently

**3. Install dependencies**
The first cell in the notebook installs everything:
```python
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install fvcore iopath
!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py312_cu121_pyt280/download.html
!pip install opencv-python matplotlib numpy trimesh
```

**4. Run the notebook**
- **Runtime → Run all**
- Downloads ShapeNet models, renders views, and performs reconstruction

---

#### Option 2: Real-World Photos (Extended Pipeline)

**1. Open the real photos notebook**
- Select: `notebooks/real_photos_reconstruction.ipynb`

**2. Take photos of an object**
- Use phone camera
- 15-20 photos from different angles around the object
- Keep the object in the center of the frame
- Consistent lighting
- Avoid motion blur

**3. Upload photos to Colab**
```python
from google.colab import files
uploaded = files.upload()
```

**4. Run reconstruction**
- Notebook handles feature detection, matching, and triangulation automatically
- Output saved to `outputs/real_object_views/`

---

#### Option 3: Interactive Web Viewer

**1. Open the web viewer notebook**
- Select: `notebooks/web_viewer_setup.ipynb`

**2. Generate web-compatible data**
- Converts PLY point clouds to JSON format
- Creates Three.js viewer HTML

**3. View locally or deploy**
- Open `web_viewer/index.html` in browser for local viewing
- Deploy to GitHub Pages for online sharing

---

## Pipeline Overview

### Part 1: ShapeNet Synthetic Data (Jorge)

**Step 1: Data Loading**
- Download ShapeNet chairs (`03001627.zip`) and cars (`02958343.zip`)
- Extract single model for reconstruction

**Step 2: Multi-View Rendering**
- Use PyTorch3D to render 8 views at different azimuths (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
- Camera distance: 1.0-1.2 units
- Resolution: 512x512 pixels

**Step 3: Feature Detection & Matching**
- SIFT keypoint detection on each view
- Brute-force matcher with Lowe's ratio test (threshold: 0.75)

**Step 4: Pose Estimation**
- Compute essential matrix using RANSAC
- Recover rotation (R) and translation (t) between cameras

**Step 5: 3D Reconstruction**
- Triangulate matched points using camera projection matrices
- Generate sparse 3D point cloud
- Export to PLY format

---

### Part 2: Real-World Photos (Kristian)

**Step 1: Photo Capture**
- Capture 15-20 images of real object with phone camera
- Maintain consistent distance and lighting

**Step 2: Feature Detection & Matching**
- Apply same SIFT pipeline to real photos
- Match features across image pairs

**Step 3: Pose Estimation & Reconstruction**
- Estimate camera poses from feature correspondences
- Triangulate to generate 3D point cloud

**Step 4: Web Visualization**
- Convert point cloud to web-compatible format
- Deploy interactive Three.js viewer

---

## Results

### ShapeNet Chair Reconstruction
- **Input**: 8 rendered views of ShapeNet chair model
- **Output**: 13-point sparse 3D reconstruction
- **File**: `outputs/chair_reconstruction.ply`

### ShapeNet Car Reconstruction
- **Input**: 8 rendered views of ShapeNet car model (52,081 vertices)
- **Output**: Colored multi-view renders with blue texture
- **File**: `outputs/car_views/`

### Real-World Object Reconstruction
- **Input**: 15-20 phone camera photos
- **Output**: Sparse 3D point cloud + interactive web viewer
- **Files**: `outputs/real_object_reconstruction.ply`, `web_viewer/index.html`

---

## Viewing Results

### In Colab (automatic)
- Point clouds display in matplotlib 3D plots
- Rendered views show in subplot grids

### In MeshLab (local)
1. Download `.ply` file from Colab
2. Open MeshLab
3. **File → Import Mesh → Select .ply file**
4. Adjust point size: **Render → Show Points (increase point size)**

### In Web Browser (interactive)
1. Open `web_viewer/index.html` in browser
2. Use mouse to rotate, zoom, pan
3. Share link if deployed to GitHub Pages

---

## Technical Details

### Key Libraries
- **PyTorch3D**: 3D rendering and mesh operations
- **OpenCV**: SIFT feature detection, matching, pose estimation
- **NumPy**: Matrix operations and numerical computing
- **Matplotlib**: Visualization
- **Trimesh**: PLY file export
- **Three.js**: Web-based 3D visualization

### Algorithms Implemented
- **SIFT** (Scale-Invariant Feature Transform) for keypoint detection
- **Lowe's ratio test** for robust feature matching
- **RANSAC** (Random Sample Consensus) for outlier rejection
- **Essential matrix decomposition** for camera pose recovery
- **Triangulation** for 3D point reconstruction

### Camera Model
- Pinhole camera with known intrinsics
- Focal length estimated from image width
- Principal point at image center

---

## Known Limitations
- **Sparse reconstruction**: Only matched keypoints are reconstructed
- **No texture mapping**: Point clouds only, no surface reconstruction
- **Limited views**: 8-20 views may be insufficient for complex objects
- **Sensitive to lighting**: Real photos require consistent illumination
- **Manual capture**: Requires careful photo positioning

### Future Improvements
- Dense reconstruction using multi-view stereo (COLMAP)
- Surface reconstruction (Poisson surface reconstruction)
- Texture mapping from original images
- Automatic photo quality validation
- NeRF-based novel view synthesis

---

## Presentation Structure

### Jorge's Section (ShapeNet Pipeline)
- Project overview and goals
- ShapeNet dataset and PyTorch3D rendering
- Feature detection and matching visualization
- Camera pose estimation theory
- Sparse reconstruction results (chair + car)

### Kristian's Section (Real-World Extension)
- Motivation for real-world testing
- Photo capture methodology
- Real object reconstruction results
- Interactive web viewer demo
- Comparison: synthetic vs. real-world challenges

---

## Deliverables
- Three Google Colab notebooks (ShapeNet, real photos, web viewer)
- 8 rendered views per ShapeNet object (chair + car)
- 15-20 real-world photos
- Feature matching visualizations
- Sparse 3D point clouds (PLY format)
- Interactive Three.js web viewer
- Final project report (3-5 pages)
- Video demonstration (Week 10)

---

## Timeline
- **Week 1-4**: ShapeNet pipeline development (Jorge)
- **Week 5-6**: Real-world photo reconstruction (Kristian)
- **Week 6-7**: Web viewer implementation (Kristian)
- **Week 8**: Integration, testing, and documentation
- **Week 9-10**: Presentation preparation and final report

---

## References
- [ShapeNet Dataset](https://shapenet.org)
- [PyTorch3D Documentation](https://pytorch3d.org/)
- [OpenCV Feature Detection Tutorial](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
- [Three.js Documentation](https://threejs.org/docs/)
- [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/) (Hartley & Zisserman)

---

## License
This project is for educational purposes as part of CS 4337 at Texas State University.

---

## Contact
- Jorge Martinez-Lopez: [GitHub](https://github.com/jorgemar723)
- Kristian Parra: [GitHub](https://github.com/KristianParra)



