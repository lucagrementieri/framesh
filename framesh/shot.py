import numpy as np
import numpy.typing as npt
import trimesh

from .util import get_connected_nearby_indices, robust_sign, round_zeros


def shot_lrf(
    mesh: trimesh.Trimesh,
    vertex_index: int,
    radius: float,
    *,
    use_vertex_normal: bool = False,
) -> npt.NDArray[np.float64]:
    """Computes a Local Reference Frame (LRF) for a vertex using the SHOT method.

    This function implements the Local Reference Frame computation from the SHOT
    (Signature of Histograms of OrienTations) descriptor. It creates a robust and
    repeatable local coordinate system at a given vertex using weighted covariance
    analysis of neighboring points.

    Args:
        mesh: The input 3D mesh.
        vertex_index: Index of the vertex for which to compute the LRF.
        radius: Support radius for the LRF computation.
        use_vertex_normal: If True, uses the vertex normal directly as the
            z-axis of the LRF. If False, computes the z-axis from covariance analysis.

    Returns:
        Axes of the LRF stored in columns [x-axis, y-axis, z-axis] forming a right-handed
        coordinate system.
        Shape: (3, 3)

    Note:
        The implementation follows these steps:
        1. Identifies neighboring points within the support radius
        2. Computes weighted covariance using distance-based weights
        3. Performs eigendecomposition to get initial axes
        4. Ensures consistent orientation using majority voting and vertex normal
        5. Returns orthonormal axes forming a right-handed coordinate system

    Reference:
        Tombari, F., Salti, S., & Di Stefano, L. (2010).
        "Unique signatures of histograms for local surface description."
        European Conference on Computer Vision (ECCV).
    """
    vertex = mesh.vertices[vertex_index]
    neighbors = get_connected_nearby_indices(mesh, vertex_index, radius, exclude_self=True)
    assert isinstance(neighbors, np.ndarray)
    if neighbors.size == 0:
        invalid_axes = np.full((3, 3), np.nan)
        if use_vertex_normal:
            invalid_axes[:, 2] = round_zeros(mesh.vertex_normals[vertex_index])
        return invalid_axes

    differences = mesh.vertices[neighbors] - vertex
    distances = trimesh.util.row_norm(differences)
    scale_factors = radius - distances
    scale_factors /= scale_factors.sum()
    weighted_covariance = round_zeros(
        np.einsum("i,ij,ik->jk", scale_factors, differences, differences)
    )
    _, eigenvectors = np.linalg.eigh(weighted_covariance)
    eigenvectors = round_zeros(eigenvectors)
    axes = np.fliplr(eigenvectors)

    x_sign_votes = robust_sign(np.dot(differences, axes[:, 0]))
    if np.sum(x_sign_votes) < 0:
        axes[:, 0] *= -1

    if use_vertex_normal:
        z_axis = round_zeros(mesh.vertex_normals[vertex_index])
        invalid_cross = np.allclose(np.cross(z_axis, axes[:, 0]), 0)
        if invalid_cross:
            axes = np.roll(axes, -1, axis=-1)
        axes[:, 2] = z_axis
        axes[:, 1] = trimesh.transformations.unit_vector(np.cross(axes[:, 2], axes[:, 0]))
        axes[:, 0] = np.cross(axes[:, 1], axes[:, 2])
    else:
        if np.dot(mesh.vertex_normals[vertex_index], axes[:, 2]) < 0.0:
            axes[:, 2] *= -1
        if np.linalg.det(axes) < 0:
            axes[:, 1] *= -1
    return axes


def shot_frames(
    mesh: trimesh.Trimesh,
    vertex_indices: npt.NDArray[np.int_],
    radius: float,
    *,
    use_vertex_normal: bool = False,
) -> npt.NDArray[np.float64]:
    """Computes Local Reference Frames (LRFs) for multiple vertices using the SHOT method.

    Vectorized version of shot_lrf that computes LRFs for multiple vertices simultaneously.

    Args:
        mesh: The input 3D mesh.
        vertex_indices: Array of vertex indices for which to compute LRFs.
            Shape: (L,) where L is the number of vertices with LRFs.
        radius: Support radius for the LRF computation.
        use_vertex_normal: If True, uses vertex normals directly as the
            z-axes of the LRFs. If False, computes z-axes from covariance analysis.

    Returns:
        Batch of axes of the LRFs stored in columns [x-axis, y-axis, z-axis] forming
        right-handed coordinate systems.
        Shape: (L, 3, 3)
    """
    vertex_indices = np.atleast_1d(vertex_indices)
    frame_vertices = mesh.vertices[vertex_indices]
    n_vertices = len(vertex_indices)

    neighbors = get_connected_nearby_indices(mesh, vertex_indices, radius, exclude_self=True)
    neighbors_counts = np.array([len(n) for n in neighbors])
    axes = np.full((n_vertices, 3, 3), np.nan)
    if use_vertex_normal:
        axes[..., 2] = round_zeros(mesh.vertex_normals[vertex_indices])

    valid_neighborhoods = neighbors_counts > 0
    if not np.any(valid_neighborhoods):
        return axes

    vertex_indices = vertex_indices[valid_neighborhoods]
    frame_vertices = frame_vertices[valid_neighborhoods]
    n_vertices = len(vertex_indices)
    neighbors_counts = neighbors_counts[valid_neighborhoods]

    flat_neighbors = np.concatenate(neighbors)
    frame_indices = np.repeat(np.arange(n_vertices), neighbors_counts)
    differences = mesh.vertices[flat_neighbors] - frame_vertices[frame_indices]  # (M, 3)
    distances = trimesh.util.row_norm(differences)
    scale_factors = radius - distances
    reduce_indices = np.insert(np.cumsum(neighbors_counts)[:-1], 0, 0)
    scale_factor_normalizer = np.add.reduceat(scale_factors, reduce_indices)
    scale_factors /= scale_factor_normalizer[frame_indices]
    covariances = np.einsum("m,mi,mj->mij", scale_factors, differences, differences)
    weighted_covariance = round_zeros(np.add.reduceat(covariances, reduce_indices))

    # Compute eigendecomposition for all vertices
    _, eigenvectors = np.linalg.eigh(weighted_covariance)
    eigenvectors = round_zeros(eigenvectors)
    valid_axes = np.flip(eigenvectors, axis=-1)

    # Ensure consistent x-axis orientation
    x_sign_votes = robust_sign(np.sum(differences * valid_axes[frame_indices, :, 0], axis=-1))
    x_sign = np.add.reduceat(x_sign_votes, reduce_indices) < 0
    valid_axes[x_sign, :, 0] *= -1
    if use_vertex_normal:
        z_axes = round_zeros(mesh.vertex_normals[vertex_indices])
        invalid_cross_mask = np.isclose(np.cross(z_axes, valid_axes[..., 0]), 0).all(axis=1)
        valid_axes[invalid_cross_mask] = np.roll(valid_axes[invalid_cross_mask], -1, axis=-1)
        valid_axes[..., 2] = z_axes
        valid_axes[..., 1] = trimesh.transformations.unit_vector(
            np.cross(valid_axes[..., 2], valid_axes[..., 0]), axis=-1
        )
        valid_axes[..., 0] = np.cross(valid_axes[..., 1], valid_axes[..., 2])
    else:
        # Ensure consistent z-axis orientation with vertex normals
        z_dots = np.sum(mesh.vertex_normals[vertex_indices] * valid_axes[..., 2], axis=-1)
        z_sign = z_dots < 0
        valid_axes[z_sign, :, 2] *= -1

        # Ensure right-handed coordinate system
        y_sign = np.linalg.det(valid_axes) < 0
        valid_axes[y_sign, :, 1] *= -1
    axes[valid_neighborhoods] = valid_axes
    return axes
