import logging
from typing import List

import fpsample.wrapper
import numpy as np
import trimesh
from scipy.spatial import KDTree


def extract_patches(mesh: trimesh.Trimesh, n_patches: int) -> List[trimesh.Trimesh]:
    n_vertices = mesh.vertices.shape[0]
    if n_patches >= n_vertices:
        logging.warning(
            "The requested number of patches is higher than the number of vertices "
            "in the mesh, a lower number of patches (one for every vertex) is returned"
        )
        center_vertex_indices = np.arange(n_vertices)
    else:
        vertices_with_centroid = np.vstack(
            [mesh.vertices, mesh.centroid[np.newaxis]]
        ).astype(np.float32)
        center_vertex_indices = fpsample.wrapper._bucket_fps_kdline_sampling(
            vertices_with_centroid,
            n_patches + 1,
            max(5, int(np.log2(n_vertices + 1))),
            n_vertices,
        )
        center_vertex_indices = np.sort(center_vertex_indices)
        assert center_vertex_indices[-1] == n_vertices
        center_vertex_indices = center_vertex_indices[:-1]
    face_centers = np.mean(mesh.vertices[mesh.faces], axis=1)
    _, face_patch_indices = KDTree(mesh.vertices[center_vertex_indices]).query(
        face_centers, workers=-1
    )
    patch_indices = []
    for patch_index, vertex_index in enumerate(center_vertex_indices):
        nearby_face_mask = face_patch_indices == patch_index
        neighbor_face_mask = np.any(np.isin(mesh.faces, vertex_index), axis=1)
        face_indices = np.flatnonzero(
            np.logical_or(nearby_face_mask, neighbor_face_mask)
        )
        patch_indices.append(face_indices)
        print(patch_indices[-1].dtype, patch_indices[-1].shape)
    patches = mesh.submesh(patch_indices, only_watertight=False, append=False)
    patch_center_vertices = mesh.vertices[center_vertex_indices]
    return patches, patch_center_vertices


if __name__ == "__main__":
    from pathlib import Path

    mesh = trimesh.load_mesh(
        Path(__file__).parents[1]
        / "common-3d-test-models"
        / "data"
        / "stanford-bunny.obj"
    )
    patch_dir = Path(__file__).parents[1] / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    for i, (patch, _) in enumerate(extract_patches(mesh, 100)):
        patch.export(patch_dir / f"patch_{i:03d}.ply")
