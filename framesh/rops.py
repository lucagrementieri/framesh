import numpy as np
import numpy.typing as npt
import trimesh

from .util import get_connected_nearby_face_indices, round_zeros


def rops_lrf(
    mesh: trimesh.Trimesh,
    vertex_index: int,
    radius: float,
    *,
    use_vertex_normal: bool = False,
) -> npt.NDArray[np.float64]:
    """Computes the Local Reference Frame (LRF) for a vertex using Rotational Projection Statistics.

    This function implements the LRF computation method described in the paper
    "A local feature descriptor for 3D rigid objects based on rotational projection statistics"
    (ICCSPA 2013). The LRF provides a robust coordinate system for local feature description.

    Args:
        mesh: The input 3D mesh.
        vertex_index: Index of the vertex for which to compute the LRF.
        radius: Support radius for the LRF computation.
        use_vertex_normal: If True, uses the vertex normal directly as the
            z-axis of the LRF. If False, computes the z-axis from scatter matrix analysis.

    Returns:
        Axes of the LRF stored in columns [x-axis, y-axis, z-axis] forming a right-handed
        coordinate system.
        Shape: (3, 3)

    Note:
        The implementation follows these steps:
        1. Computes a weighted scatter matrix using face areas and distances
        2. Performs eigendecomposition to get initial axes
        3. Ensures consistent orientation using vertex normal and projection signs
        4. Returns orthonormal axes forming a right-handed coordinate system

    Reference:
        Guo, Y., Sohel, F. A., Bennamoun, M., Wan, J., & Lu, M. (2013).
        "A local feature descriptor for 3D rigid objects based on rotational projection statistics."
        International Conference on Communications, Signal Processing, and their Applications
        (ICCSPA).
    """
    vertex = mesh.vertices[vertex_index]

    local_triangle_indices = get_connected_nearby_face_indices(mesh, vertex_index, radius)
    differences = mesh.triangles[local_triangle_indices] - vertex
    area_weights = mesh.area_faces[local_triangle_indices]
    area_weights /= area_weights.sum()
    centers_differences = mesh.triangles_center[local_triangle_indices] - vertex
    distance_weights = np.square(
        np.clip(radius - trimesh.util.row_norm(centers_differences), a_min=0, a_max=radius)
    )

    mesh_scatter = round_zeros(
        np.einsum(
            "fik,fjm,ij,f->km",
            differences,
            differences,
            np.eye(3) + 1,
            area_weights * distance_weights,
            optimize=True,
        )
        / 12
    )
    _, eigenvectors = np.linalg.eigh(mesh_scatter)
    eigenvectors = round_zeros(eigenvectors)
    axes = np.fliplr(eigenvectors)

    hx = np.einsum(
        "fk,k,f->",
        centers_differences,
        axes[:, 0],
        area_weights * distance_weights,
        optimize=True,
    )
    if hx < 0:
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
        axes[:, 1] = np.cross(axes[:, 2], axes[:, 0])
    return axes


def rops_frames(
    mesh: trimesh.Trimesh,
    vertex_indices: npt.NDArray[np.int_],
    radius: float,
    *,
    use_vertex_normal: bool = False,
) -> npt.NDArray[np.float64]:
    """Computes Local Reference Frames (LRFs) for multiple vertices using RoPS method.

    Vectorized version of rops_lrf that computes LRFs for multiple vertices simultaneously.

    Args:
        mesh: The input 3D mesh.
        vertex_indices: Array of vertex indices for which to compute LRFs.
            Shape: (L,) where L is the number of vertices with LRFs.
        radius: Support radius for the LRF computation.
        use_vertex_normal: If True, uses vertex normals directly as the
            z-axes of the LRFs. If False, computes z-axes from scatter matrix analysis.

    Returns:
        Batch of axes of the LRFs stored in columns [x-axis, y-axis, z-axis] forming
        right-handed coordinate systems.
        Shape: (L, 3, 3)
    """
    vertex_indices = np.atleast_1d(vertex_indices)
    n_vertices = len(vertex_indices)

    local_triangle_indices = get_connected_nearby_face_indices(mesh, vertex_indices, radius)
    triangles_counts = np.array(
        [len(triangle_indices) for triangle_indices in local_triangle_indices]
    )
    flat_triangle_indices = np.concatenate(local_triangle_indices)
    frame_indices = np.repeat(np.arange(n_vertices), triangles_counts)
    reduce_indices = np.insert(np.cumsum(triangles_counts)[:-1], 0, 0)
    differences = mesh.triangles[flat_triangle_indices] - np.expand_dims(
        mesh.vertices[vertex_indices[frame_indices]], axis=1
    )
    area_weights = mesh.area_faces[flat_triangle_indices]
    area_weight_normalizer = np.add.reduceat(area_weights, reduce_indices)
    area_weights /= area_weight_normalizer[frame_indices]
    centers_differences = (
        mesh.triangles_center[flat_triangle_indices] - mesh.vertices[vertex_indices[frame_indices]]
    )
    distance_weights = np.square(
        np.clip(radius - trimesh.util.row_norm(centers_differences), a_min=0, a_max=radius)
    )
    weights = area_weights * distance_weights

    # Compute scatter matrix with diagonal adjustment
    mesh_scatter = np.empty((n_vertices, 3, 3))
    for frame_index in range(n_vertices):
        mask = frame_indices == frame_index
        mesh_scatter[frame_index] = round_zeros(
            np.einsum(
                "sik,sjm,ij,s->km",
                differences[mask],
                differences[mask],
                np.eye(3) + 1,
                weights[mask],
                optimize=True,
            )
            / 12
        )

    # Compute eigendecomposition for all vertices
    _, eigenvectors = np.linalg.eigh(mesh_scatter)
    eigenvectors = round_zeros(eigenvectors)
    axes = np.flip(eigenvectors, axis=-1)

    # Ensure consistent x-axis orientation
    triangle_hx = np.einsum(
        "mk,mk,m->m",
        centers_differences,
        axes[frame_indices, :, 0],
        area_weights * distance_weights,
        optimize=True,
    )
    hx = np.add.reduceat(triangle_hx, reduce_indices)
    x_sign = hx < 0
    axes[x_sign, :, 0] *= -1

    if use_vertex_normal:
        z_axes = round_zeros(mesh.vertex_normals[vertex_indices])
        invalid_cross_mask = np.isclose(np.cross(z_axes, axes[..., 0]), 0).all(axis=1)
        axes[invalid_cross_mask] = np.roll(axes[invalid_cross_mask], -1, axis=-1)
        axes[..., 2] = z_axes
        axes[..., 1] = trimesh.transformations.unit_vector(
            np.cross(axes[..., 2], axes[..., 0]), axis=-1
        )
        axes[..., 0] = np.cross(axes[..., 1], axes[..., 2])
    else:
        # Ensure consistent z-axis orientation with vertex normals
        z_dots = np.sum(mesh.vertex_normals[vertex_indices] * axes[..., 2], axis=-1)
        z_sign = z_dots < 0
        axes[z_sign, ..., 2] *= -1
        axes[..., 1] = np.cross(axes[..., 2], axes[..., 0])

    return axes


def rops_frames_iterative(
    mesh: trimesh.Trimesh,
    vertex_indices: npt.NDArray[np.int_],
    radius: float,
    *,
    use_vertex_normal: bool = False,
) -> npt.NDArray[np.float64]:
    """Computes Local Reference Frames (LRFs) for multiple vertices using RoPS method.

    Simple iterative version of rops_frames that computes LRFs for multiple vertices
    by calling rops_lrf repeatedly. It is included here because sometimes it is faster than
    rops_frames.

    Args:
        mesh: The input 3D mesh.
        vertex_indices: Array of vertex indices for which to compute LRFs.
            Shape: (L,) where L is the number of vertices with LRFs.
        radius: Support radius for the LRF computation.
        use_vertex_normal: If True, uses vertex normals directly as the
            z-axes of the LRFs. If False, computes z-axes from scatter matrix analysis.

    Returns:
        Batch of axes of the LRFs stored in columns [x-axis, y-axis, z-axis] forming
        right-handed coordinate systems.
        Shape: (L, 3, 3)
    """
    lrf = np.stack(
        [rops_lrf(mesh, v, radius, use_vertex_normal=use_vertex_normal) for v in vertex_indices]
    )
    return lrf
