from typing import Optional

import igl
import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.sparse.linalg
import trimesh

from .util import get_nearby_indices, timeit


def fiedler_squared(mesh):
    mass_diagonal = igl.massmatrix(mesh.vertices, mesh.faces).diagonal()
    laplacian = -igl.cotmatrix(mesh.vertices, mesh.faces) / mass_diagonal
    w, v = scipy.sparse.linalg.eigsh(laplacian, k=2, sigma=0)
    return np.square(v[:, 1])


def gaussian_curvature(mesh: trimesh.Trimesh, eps: float = 1e-14) -> npt.NDArray[np.float64]:
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

    area_mixed = igl.massmatrix(mesh.vertices, mesh.faces).diagonal()
    curvature = np.divide(
        defects, area_mixed, out=np.zeros_like(area_mixed), where=area_mixed > eps
    )
    return curvature


def mean_curvature(mesh: trimesh.Trimesh, eps: float = 1e-14) -> npt.NDArray[np.float64]:
    laplacian = igl.cotmatrix(mesh.vertices, mesh.faces)
    position_laplacian = laplacian.dot(mesh.vertices)
    unscaled_curvature = trimesh.util.row_norm(position_laplacian)
    area_mixed = igl.massmatrix(mesh.vertices, mesh.faces).diagonal()
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
    use_vertex_normal: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Reference: Gframes: Gradient-based local reference frame for 3d shape matching. (CVPR 2019)
    Authors: Simone Melzi, Riccardo Spezialetti, Federico Tombari, Michael M. Bronstein, Luigi Di Stefano, and Emanuele Rodola.
    """
    z_axis = mesh.vertex_normals[vertex_index]

    x_neighbors = get_nearby_indices(mesh, vertex_index, radius)
    x_vertices = mesh.vertices[x_neighbors]
    differences = x_vertices - vertex
    distances = trimesh.util.row_norm(differences)
    projection_distances = np.dot(differences, z_axis)
    scale_factors = np.square((radius - distances) * projection_distances)
    x_axis = np.dot(x_vertices.T, scale_factors)
    y_axis = trimesh.transformations.unit_vector(np.cross(z_axis, x_axis))
    x_axis = np.cross(y_axis, z_axis)
    axes = np.column_stack((x_axis, y_axis, z_axis))
    return axes
