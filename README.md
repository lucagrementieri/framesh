# 🖼️ Framesh 

A collection of robust Local Reference Frame (LRF) algorithms for 3D mesh patches.

## Overview

Framesh provides efficient implementations of several state-of-the-art Local Reference Frame 
computation methods for 3D meshes. LRFs are essential for creating rotation-invariant local surface 
descriptors and establishing repeatable coordinate systems on 3D surfaces. The library has minimal 
dependencies, requiring only NumPy for computation and Trimesh for mesh handling.

## Installation

Framesh is available on PyPI and can be installed using pip:

```bash
python -m pip install framesh
``` 

## Usage

Each LRF method follows a consistent API:

```python
import trimesh
from framesh import shot_lrf

# Load your mesh
mesh = trimesh.load('your_mesh.obj')
# Compute LRF for vertex index 0 with default parameters
lrf = shot_lrf(mesh, vertex_index=0, radius=1.0)
# The result is a 3x3 matrix where columns are [x-axis, y-axis, z-axis]
print(lrf)
```

## Implemented Methods

- **SHOT**
  - Implementation of the LRF computation from the SHOT (Signature of Histograms of 
    OrienTations) descriptor. Uses weighted covariance analysis to establish a robust 
    coordinate system.
  - *Reference:* Tombari et al., "Unique signatures of histograms for local surface 
    description." ECCV 2010.

- **BOARD**
  - Board method for LRF computation. Creates a robust coordinate system using plane fitting 
    and normal-based point selection strategies.
  - *Reference:* Petrelli & Di Stefano, "On the repeatability of the local reference frame 
    for partial shape matching." ICCV 2011.

- **FLARE**
  - Fast Local Axis Reference Extraction method. Combines plane fitting for z-axis computation 
    with a distance-based point selection strategy for x-axis determination.
  - *Reference:* Petrelli & Di Stefano, "A Repeatable and Efficient Canonical Reference for 
    Surface Matching." 3DIMPVT 2012.

- **ROPS**
  - Rotational Projection Statistics-based LRF computation. Utilizes face areas and distances 
    to construct a weighted scatter matrix for axis determination.
  - *Reference:* Guo et al., "A local feature descriptor for 3D rigid objects based on 
    rotational projection statistics." ICCSPA 2013.

- **TOLDI**
  - Triangular-based Overlapping Local Depth Images LRF computation. Uses a combination of PCA 
    and projection-based weighting for robust axis determination.
  - *Reference:* Yang et al., "TOLDI: An effective and robust approach for 3D local shape 
    description." Pattern Recognition 2017.

- **GFrames**
  - Gradient-based local reference frame computation. Utilizes the gradient of a scalar field 
    defined on the mesh surface to determine axis directions.
  - *Reference:* Melzi et al., "GFrames: Gradient-based local reference frame for 3D shape 
    matching." CVPR 2019.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{grementieri2024framesh,
    author = {Grementieri, Luca},
    title = {Framesh: A Collection of Local Reference Frame Algorithms for 3D Mesh Patches},
    year = {2024},
    publisher = {GitHub},
    url = {https://github.com/lucagrementieri/framesh},
    note = {A library implementing multiple classical Local Reference Frame (LRF) algorithms}
}
```
