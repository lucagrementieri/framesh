import numpy as np
import numpy.typing as npt
import trimesh

from .util import get_nearby_indices, round_zeros


def board_lrf(
    mesh: trimesh.Trimesh,
    vertex_index: int,
    radius: float | None = None,
    *,
    use_vertex_normal: bool = False,
    z_radius: float | None = None,
) -> npt.NDArray[np.float64]:
    """Computes a Local Reference Frame (LRF) for a vertex using the Board method.

    This function implements the LRF computation from the Board method, which creates a robust
    local coordinate system at a given vertex using plane fitting and normal-based point selection.

    Args:
        mesh: The input 3D mesh.
        vertex_index: Index of the vertex for which to compute the LRF.
        radius: Support radius for the LRF computation. If None,
            uses the maximum distance from the vertex to any other vertex.
        use_vertex_normal: If True, uses the vertex normal directly as the
            z-axis of the LRF. If False, computes the z-axis from plane fitting.
        z_radius: Support radius for z-axis computation. If None,
            uses the same value as radius.

    Returns:
        Axes of the LRF stored in columns [x-axis, y-axis, z-axis] forming a right-handed
        coordinate system.
        Shape: (3, 3)

    Note:
        The implementation follows these steps:
        1. Computes z-axis by plane fitting or using vertex normal
        2. Selects points outside 85% of support radius
        3. Finds point with normal most perpendicular to z-axis
        4. Uses this point to define x-axis direction
        5. Completes right-handed coordinate system

    Reference:
        Petrelli, A., & Di Stefano, L. (2011).
        "On the repeatability of the local reference frame for partial shape matching."
        International Conference on Computer Vision (ICCV).
    """
    vertex = mesh.vertices[vertex_index]

    if z_radius is None:
        z_radius = radius
    if use_vertex_normal:
        z_neighbors = None
        z_axis = round_zeros(mesh.vertex_normals[vertex_index])
    else:
        z_neighbors = get_nearby_indices(mesh, vertex_index, z_radius)
        _, z_axis = trimesh.points.plane_fit(mesh.vertices[z_neighbors])
        z_axis = round_zeros(z_axis)
        if np.dot(z_axis, mesh.vertex_normals[z_neighbors].sum(axis=0)) < 0.0:
            z_axis *= -1

    if z_neighbors is not None and radius == z_radius:
        x_neighbors = z_neighbors
    else:
        x_neighbors = get_nearby_indices(mesh, vertex_index, radius)
    distances = trimesh.util.row_norm(mesh.vertices[x_neighbors] - vertex)
    EXCLUDE_RADIUS_COEFFICIENT = 0.85
    exclude_radius = EXCLUDE_RADIUS_COEFFICIENT * (np.max(distances) if radius is None else radius)
    radius_mask = distances > exclude_radius
    if np.any(radius_mask):
        x_neighbors = x_neighbors[distances > exclude_radius]
    x_point_index = np.argmin(np.abs(np.dot(mesh.vertex_normals[x_neighbors], z_axis)))
    x_vector = mesh.vertices[x_neighbors[x_point_index]] - vertex
    y_axis = trimesh.transformations.unit_vector(np.cross(z_axis, x_vector))
    x_axis = np.cross(y_axis, z_axis)
    axes = np.column_stack((x_axis, y_axis, z_axis))
    return axes


def board_frames(
    mesh: trimesh.Trimesh,
    vertex_indices: npt.NDArray[np.int_],
    radius: float | None = None,
    *,
    use_vertex_normal: bool = False,
    z_radius: float | None = None,
) -> npt.NDArray[np.float64]:
    """Computes Local Reference Frames (LRFs) for multiple vertices using the Board method.

    Vectorized version of board_lrf that computes LRFs for multiple vertices simultaneously.

    Args:
        mesh: The input 3D mesh.
        vertex_indices: Array of vertex indices for which to compute LRFs.
            Shape: (L,) where L is the number of vertices with LRFs.
        radius: Support radius for the LRF computation. If None,
            uses the maximum distance from each vertex to any other vertex.
        use_vertex_normal: If True, uses vertex normals directly as the
            z-axes of the LRFs. If False, computes z-axes from plane fitting.
        z_radius: Support radius for z-axis computation. If None,
            uses the same value as radius.

    Returns:
        Batch of axes of the LRFs stored in columns [x-axis, y-axis, z-axis] forming
        right-handed coordinate systems.
        Shape: (L, 3, 3)
    """
    vertex_indices = np.atleast_1d(vertex_indices)
    frame_vertices = mesh.vertices[vertex_indices]
    n_vertices = len(vertex_indices)

    if z_radius is None:
        z_radius = radius
    if use_vertex_normal:
        z_neighbors = None
        z_axes = round_zeros(mesh.vertex_normals[vertex_indices])
    elif z_radius is None:
        _, z_axis = trimesh.points.plane_fit(mesh.vertices)
        z_axis = round_zeros(z_axis)
        if np.dot(z_axis, mesh.vertex_normals.sum(axis=0)) < 0.0:
            z_axis *= -1
        z_axes = np.tile(z_axis, (len(vertex_indices), 1))
    else:
        z_neighbors = get_nearby_indices(mesh, vertex_indices, z_radius)
        z_axes = np.zeros((n_vertices, 3))
        for i, neighbors in enumerate(z_neighbors):
            _, z_axes[i] = trimesh.points.plane_fit(mesh.vertices[neighbors])
            z_axes[i] = round_zeros(z_axes[i])
            if np.dot(z_axes[i], mesh.vertex_normals[neighbors].sum(axis=0)) < 0.0:
                z_axes[i] *= -1

    EXCLUDE_RADIUS_COEFFICIENT = 0.85
    if radius is None:
        # Use matrix operations since neighbors have all the same size
        differences = mesh.vertices - np.expand_dims(frame_vertices, axis=1)  # (L, V, 3)
        distances = np.linalg.norm(differences, axis=-1)  # (L, V)
        exclude_radius = EXCLUDE_RADIUS_COEFFICIENT * np.max(distances, axis=-1, keepdims=True)
        frame_indices, x_neighbors = np.nonzero(distances > exclude_radius)
    else:
        if z_neighbors is not None and radius == z_radius:
            x_neighbors = z_neighbors
        else:
            x_neighbors = get_nearby_indices(mesh, vertex_indices, radius)
        neighbors_counts = np.array([len(n) for n in x_neighbors])
        flat_neighbors = np.concatenate(x_neighbors)
        frame_indices = np.repeat(np.arange(n_vertices), neighbors_counts)
        differences = mesh.vertices[flat_neighbors] - frame_vertices[frame_indices]  # (M, 3)
        distances = trimesh.util.row_norm(differences)
        exclude_radius = EXCLUDE_RADIUS_COEFFICIENT * radius
        selected_indices = np.flatnonzero(distances > exclude_radius)
        valid_frame_indices = trimesh.grouping.unique_bincount(frame_indices[selected_indices])
        if not np.array_equal(valid_frame_indices, np.arange(n_vertices)):
            exclude_radius[np.isin(frame_indices, valid_frame_indices, invert=True)] = 0.0
            selected_indices = np.flatnonzero(distances > exclude_radius)
        frame_indices = frame_indices[selected_indices]
        x_neighbors = flat_neighbors[selected_indices]

    unique_frame_indices, x_neighbors_counts = trimesh.grouping.unique_bincount(
        frame_indices, return_counts=True
    )
    assert np.array_equal(unique_frame_indices, np.arange(n_vertices))
    assert np.array_equal(frame_indices, np.repeat(np.arange(n_vertices), x_neighbors_counts))
    reduce_indices = np.insert(np.cumsum(x_neighbors_counts)[:-1], 0, 0)
    z_dots = np.abs(np.sum(mesh.vertex_normals[x_neighbors] * z_axes[frame_indices], axis=-1))
    x_point_value = np.minimum.reduceat(z_dots, reduce_indices)
    x_point_mask = x_point_value[frame_indices] == z_dots
    x_point_index = np.flatnonzero(x_point_mask)
    if x_point_index.size > n_vertices:
        x_point_index = x_point_index[np.unique(frame_indices[x_point_index], return_index=True)[1]]
    x_vector = mesh.vertices[x_neighbors[x_point_index]] - frame_vertices
    y_axes = trimesh.transformations.unit_vector(np.cross(z_axes, x_vector), axis=-1)
    x_axes = np.cross(y_axes, z_axes)
    axes = np.stack((x_axes, y_axes, z_axes), axis=-1)
    return axes
