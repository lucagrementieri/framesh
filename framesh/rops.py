import numpy as np
import numpy.typing as npt
import trimesh


def rops_lrf(
    mesh: trimesh.Trimesh,
    vertex_index: int,
    radius: float | None = None,
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
        radius: Support radius for the LRF computation. If None,
            uses the maximum distance from the vertex to any other vertex.
        use_vertex_normal: If True, uses the vertex normal directly as the
            z-axis of the LRF. If False, computes the z-axis from scatter matrix analysis.

    Returns:
        A 3x3 matrix where each column represents an axis of the LRF.
        The columns are [x-axis, y-axis, z-axis] forming a right-handed coordinate system.

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
    differences = mesh.vertices - mesh.vertices[vertex_index]
    distances = trimesh.util.row_norm(differences)
    if radius is None:
        radius = np.max(distances)
    area_weights = mesh.area_faces / mesh.area
    centers_differences = mesh.triangles_center - mesh.vertices[vertex_index]
    distance_weights = np.square(radius - trimesh.util.row_norm(centers_differences))

    # Compute scatter matrix with diagonal adjustment
    mesh_scatter = (
        np.einsum(
            "fik,fjm,ij,f->km",
            differences[mesh.faces],
            differences[mesh.faces],
            np.eye(3) + 1,
            area_weights * distance_weights,
            optimize=True,
        )
        / 12
    )
    _, eigenvectors = np.linalg.eigh(mesh_scatter)
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
        axes[:, 2] = mesh.vertex_normals[vertex_index]
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
    radius: float | None = None,
    *,
    use_vertex_normal: bool = False,
) -> npt.NDArray[np.float64]:
    """Computes Local Reference Frames (LRFs) for multiple vertices using RoPS method.

    Vectorized version of rops_lrf that computes LRFs for multiple vertices simultaneously.

    Args:
        mesh: The input 3D mesh.
        vertex_indices: Array of vertex indices for which to compute LRFs.
            Shape: (L,) where L is the number of vertices with LRFs.
        radius: Support radius for the LRF computation. If None,
            uses the maximum distance from each vertex to any other vertex.
        use_vertex_normal: If True, uses vertex normals directly as the
            z-axes of the LRFs. If False, computes z-axes from scatter matrix analysis.

    Returns:
        Batch of axes of the LRFs stored in columns [x-axis, y-axis, z-axis] forming
        right-handed coordinate systems.
        Shape: (L, 3, 3)
    """
    vertex_indices = np.atleast_1d(vertex_indices)
    frame_vertices = mesh.vertices[vertex_indices]
    n_vertices = len(vertex_indices)

    # Compute differences and distances for radius determination
    differences = mesh.vertices - np.expand_dims(frame_vertices, axis=1)  # (L, V, 3)
    distances = np.linalg.norm(differences, axis=-1)  # (L, V)

    radius = np.max(distances, axis=-1) if radius is None else np.repeat(radius, n_vertices)  # (L,)

    # Compute weights
    area_weights = mesh.area_faces / mesh.area  # (F,)
    face_centers = np.mean(mesh.vertices[mesh.faces], axis=1)  # (F, 3)
    centers_differences = face_centers - np.expand_dims(frame_vertices, axis=1)  # (L, F, 3)
    distance_weights = np.square(
        np.expand_dims(radius, axis=-1) - np.linalg.norm(centers_differences, axis=-1)
    )  # (L, F)

    # Compute scatter matrix with diagonal adjustment
    mesh_scatter = (
        np.einsum(
            "lfik,lfjm,ij,lf->lkm",
            differences[:, mesh.faces],
            differences[:, mesh.faces],
            np.eye(3) + 1,
            area_weights * distance_weights,
            optimize=True,
        )
        / 12
    )

    # Compute eigendecomposition for all vertices
    _, eigenvectors = np.linalg.eigh(mesh_scatter)
    axes = np.flip(eigenvectors, axis=-1)

    # Ensure consistent x-axis orientation
    hx = np.einsum(
        "lfk,lk,lf->l",
        centers_differences,
        axes[..., 0],
        area_weights * distance_weights,
        optimize=True,
    )
    x_sign = hx < 0
    axes[x_sign, :, 0] *= -1

    if use_vertex_normal:
        axes[..., 2] = mesh.vertex_normals[vertex_indices]
        axes[..., 1] = trimesh.transformations.unit_vector(
            np.cross(axes[..., 2], axes[..., 0]), axis=-1
        )
        axes[..., 0] = np.cross(axes[..., 1], axes[..., 2])
    else:
        # Ensure consistent z-axis orientation with vertex normals
        z_dots = np.einsum(
            "li,li->l",
            mesh.vertex_normals[vertex_indices],
            axes[..., 2],
            optimize=True,
        )
        z_sign = z_dots < 0
        axes[z_sign, ..., 2] *= -1
        axes[..., 1] = np.cross(axes[..., 2], axes[..., 0])

    return axes
