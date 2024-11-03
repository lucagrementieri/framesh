from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.sparse.linalg
import trimesh

from .util import get_nearby_indices, timeit


def face_half_cotangent(mesh: trimesh.Trimesh) -> npt.NDArray[np.float64]:
    half_cotangent = np.cos(mesh.face_angles) / (2 * np.sin(mesh.face_angles))
    half_cotangent[np.isclose(mesh.face_angles, 0.5 * np.pi, atol=1e-15)] = 0.0
    return half_cotangent


def cotangent_matrix(mesh: trimesh.Trimesh) -> scipy.sparse.csr_array:
    """
    Reference: https://mobile.rodolphe-vaillant.fr/entry/20/compute-harmonic-weights-on-a-triangular-mesh
    """
    cot_entries = face_half_cotangent(mesh)
    cotangent_coo = scipy.sparse.coo_array(
        (
            cot_entries[:, [2, 0, 1]].ravel(),
            tuple(mesh.edges_unique[mesh.faces_unique_edges.ravel()].T),
        ),
        shape=(len(mesh.vertices), len(mesh.vertices)),
    )
    cotangent_coo += cotangent_coo.T
    cotangent_coo.setdiag(-np.sum(cotangent_coo, axis=1))
    return cotangent_coo.tocsr()


def mass_diagonal(mesh: trimesh.Trimesh, method: str = "mixed_voronoi") -> npt.NDArray[np.float64]:
    if method == "barycentric":
        return mass_diagonal_barycentric(mesh)
    elif method == "mixed_voronoi":
        return mass_diagonal_mixed_voronoi(mesh)
    else:
        raise ValueError(
            f"Unknown mass method {method}, it should be 'barycentric' or 'mixed_voronoi'"
        )


def mass_diagonal_barycentric(mesh: trimesh.Trimesh) -> npt.NDArray[np.float64]:
    """
    Reference: https://mobile.rodolphe-vaillant.fr/entry/20/compute-harmonic-weights-on-a-triangular-mesh
    """
    vertex_areas = np.where(
        mesh.vertex_faces == -1,
        np.zeros_like(mesh.vertex_faces, dtype=np.float64),
        mesh.area_faces[mesh.vertex_faces],
    )
    return np.sum(vertex_areas, axis=1) / 3.0


def mass_diagonal_mixed_voronoi(mesh: trimesh.Trimesh) -> npt.NDArray[np.float64]:
    """
    Reference: https://mobile.rodolphe-vaillant.fr/entry/20/compute-harmonic-weights-on-a-triangular-mesh
    """
    cot_entries = face_half_cotangent(mesh)
    squared_edge_lengths = np.square(mesh.edges_unique_length[mesh.faces_unique_edges])
    area_elements = (squared_edge_lengths * cot_entries[:, [2, 0, 1]]) / 4.0
    vertex_triangle_areas = area_elements[:, [2, 0, 1]] + area_elements
    obtuse_angle_mask = cot_entries < 0
    obtuse_triangle_mask = np.any(obtuse_angle_mask, axis=1)
    vertex_triangle_areas[obtuse_triangle_mask] = np.expand_dims(
        mesh.area_faces[obtuse_triangle_mask], axis=-1
    ) / np.where(obtuse_angle_mask[obtuse_triangle_mask], 2.0, 4.0)
    vertex_areas = np.zeros_like(mesh.vertices[:, 0])
    np.add.at(vertex_areas, mesh.faces, vertex_triangle_areas)
    return vertex_areas


def fiedler_squared(mesh: trimesh.Trimesh, mass_method: str = "mixed_voronoi"):
    sparse_mass = scipy.sparse.diags(mass_diagonal(mesh, mass_method), format="csr")
    _, v = scipy.sparse.linalg.eigsh(-cotangent_matrix(mesh), M=sparse_mass, k=2, sigma=0)
    field = np.square(v[:, 1])
    scaled_field = (field - np.min(field)) / (np.max(field) - np.min(field))
    return scaled_field


def gaussian_curvature(mesh: trimesh.Trimesh, eps: float = 1e-14) -> npt.NDArray[np.float64]:
    """
    Reference: https://rodolphe-vaillant.fr/entry/33/curvature-of-a-triangle-mesh-definition-and-computation
    """
    boundary_edge_indices = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    boundary_edges = mesh.edges[boundary_edge_indices]
    sorted_boundary_edges = boundary_edges[np.lexsort(np.rot90(boundary_edges))]
    next_boundary_edges = sorted_boundary_edges[
        np.searchsorted(sorted_boundary_edges[:, 0], boundary_edges[:, 1])
    ]
    boundary_vector = np.squeeze(np.diff(mesh.vertices[np.fliplr(boundary_edges)], axis=1), axis=1)
    next_boundary_vector = np.squeeze(np.diff(mesh.vertices[next_boundary_edges], axis=1), axis=1)
    angles = trimesh.transformations.angle_between_vectors(
        boundary_vector, next_boundary_vector, axis=1
    )
    defects = np.copy(mesh.vertex_defects)
    defects[boundary_edges[:, 1]] -= angles

    area_mixed = mass_diagonal(mesh)
    curvature = np.divide(
        defects, area_mixed, out=np.zeros_like(area_mixed), where=area_mixed > eps
    )
    return curvature


def mean_curvature(mesh: trimesh.Trimesh, eps: float = 1e-14) -> npt.NDArray[np.float64]:
    """
    Reference: https://rodolphe-vaillant.fr/entry/33/curvature-of-a-triangle-mesh-definition-and-computation
    """
    laplacian = cotangent_matrix(mesh)
    position_laplacian = laplacian.dot(mesh.vertices)
    unscaled_curvature = trimesh.util.row_norm(position_laplacian)
    area_mixed = mass_diagonal(mesh)
    curvature_sign_dot = -np.sum(position_laplacian * mesh.vertex_normals, axis=-1)
    curvature_sign = np.sign(
        curvature_sign_dot,
        out=np.zeros_like(curvature_sign_dot),
        where=np.abs(curvature_sign_dot) > eps,
    )
    curvature = curvature_sign * np.divide(
        unscaled_curvature, 2 * area_mixed, out=np.zeros_like(area_mixed), where=area_mixed > eps
    )
    return curvature


@timeit
def gframes_lrf(
    mesh: trimesh.Trimesh,
    vertex_index: int,
    radius: Optional[float] = None,
    use_vertex_normal: bool = True,
    *,
    scalar_field: npt.NDArray[np.float64],
    triangle_selection_method: str = "all",
) -> npt.NDArray[np.float64]:
    """
    Reference: Gframes: Gradient-based local reference frame for 3d shape matching. (CVPR 2019)
    Authors: Simone Melzi, Riccardo Spezialetti, Federico Tombari, Michael M. Bronstein, Luigi Di Stefano, and Emanuele Rodola.
    """
    if not use_vertex_normal:
        raise ValueError("GFrames always uses the vertex normal")
    z_axis = mesh.vertex_normals[vertex_index]

    x_neighbors = get_nearby_indices(mesh, vertex_index, radius)
    if triangle_selection_method == "all":
        triangle_indices = np.flatnonzero(np.all(np.isin(mesh.faces, x_neighbors), axis=1))
    elif triangle_selection_method == "any":
        triangle_indices = np.unique(mesh.vertex_faces[x_neighbors])
        triangle_indices = triangle_indices[1:] if triangle_indices[0] == -1 else triangle_indices
    else:
        raise ValueError(
            f"Invalid triangle selection method {triangle_selection_method}: "
            "it should be one of 'all' or 'any'"
        )
    e_coefficients = mesh.edges_unique_length[mesh.faces_unique_edges[triangle_indices, 0]]
    g_coefficients = mesh.edges_unique_length[mesh.faces_unique_edges[triangle_indices, -1]]
    f_coefficients = e_coefficients * g_coefficients * np.cos(mesh.face_angles[triangle_indices, 0])
    determinants = e_coefficients * g_coefficients - np.square(f_coefficients)
    inverse_matrices = (
        np.array([[g_coefficients, -f_coefficients], [-f_coefficients, e_coefficients]])
        / determinants
    )
    inverse_matrices = np.moveaxis(inverse_matrices, -1, 0)

    triangles = mesh.faces[triangle_indices]
    edges = np.swapaxes(mesh.vertices[triangles[:, 1:]] - mesh.vertices[triangles[:, [0]]], 1, 2)
    scalar_field_differences = np.column_stack(
        [
            scalar_field[triangles[:, 1]] - scalar_field[triangles[:, 0]],
            scalar_field[triangles[:, 2]] - scalar_field[triangles[:, 0]],
        ]
    )
    triangle_areas = mesh.area_faces[triangle_indices]
    normalized_triangle_areas = triangle_areas / np.sum(triangle_areas)
    x_axis = np.einsum(
        "n,nij,njk,nk->i",
        normalized_triangle_areas,
        edges,
        inverse_matrices,
        scalar_field_differences,
    )
    y_axis = trimesh.transformations.unit_vector(np.cross(z_axis, x_axis))
    x_axis = np.cross(y_axis, z_axis)
    axes = np.column_stack((x_axis, y_axis, z_axis))
    return axes
