import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.sparse.linalg
import trimesh

from .util import get_connected_nearby_face_indices, round_zeros


def face_half_cotangent(mesh: trimesh.Trimesh) -> npt.NDArray[np.float64]:
    """Computes half cotangent values for each angle in mesh faces.

    Args:
        mesh: The input 3D mesh.

    Returns:
        Half cotangent values for each angle in the mesh faces.
        Values are set to 0 for angles very close to 90 degrees.
        Shape: (F, 3) where F is the number of faces in the mesh.
    """
    half_cotangent: npt.NDArray[np.float64] = np.cos(mesh.face_angles) / (
        2 * np.sin(mesh.face_angles)
    )
    half_cotangent[np.isclose(mesh.face_angles, 0.5 * np.pi, atol=1e-15)] = 0.0
    return half_cotangent


def cotangent_matrix(mesh: trimesh.Trimesh) -> scipy.sparse.csr_array:
    """Computes the cotangent Laplacian matrix for a mesh.

    Constructs a sparse matrix where non-diagonal entries are the sum of cotangents
    of angles opposite to each edge, and diagonal entries are negative row sums.

    Args:
        mesh: The input 3D mesh.

    Returns:
        Sparse CSR matrix containing the cotangent Laplacian weights.
        Shape: (V, V) where V is the number of vertices in the mesh.

    Reference:
        Vaillant, R.
        "Compute harmonic weights on a triangular mesh."
        https://mobile.rodolphe-vaillant.fr/entry/20/compute-harmonic-weights-on-a-triangular-mesh
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
    cotangent_coo.setdiag(-cotangent_coo.sum(axis=1))
    cotangent_csr: scipy.sparse.csr_array = cotangent_coo.tocsr()
    return cotangent_csr


def mass_diagonal(mesh: trimesh.Trimesh, method: str = "mixed_voronoi") -> npt.NDArray[np.float64]:
    """Computes the diagonal mass matrix using specified method.

    Args:
        mesh: The input 3D mesh.
        method: Method to compute mass matrix. Either 'barycentric' or 'mixed_voronoi'.

    Returns:
        Diagonal entries of the mass matrix.
        Shape: (V,) where V is the number of vertices in the mesh.

    Raises:
        ValueError: If method is not 'barycentric' or 'mixed_voronoi'.
    """
    if method == "barycentric":
        return mass_diagonal_barycentric(mesh)
    if method == "mixed_voronoi":
        return mass_diagonal_mixed_voronoi(mesh)
    raise ValueError(f"Unknown mass method {method}, it should be 'barycentric' or 'mixed_voronoi'")


def mass_diagonal_barycentric(mesh: trimesh.Trimesh) -> npt.NDArray[np.float64]:
    """Computes diagonal mass matrix using barycentric method.

    For each vertex, computes one third of the sum of areas of adjacent triangles.

    Args:
        mesh: The input 3D mesh.

    Returns:
        Diagonal entries of the barycentric mass matrix.
        Shape: (V,) where V is the number of vertices in the mesh.

    Reference:
        Vaillant, R.
        "Compute harmonic weights on a triangular mesh."
        https://mobile.rodolphe-vaillant.fr/entry/20/compute-harmonic-weights-on-a-triangular-mesh
    """
    vertex_areas = np.where(
        mesh.vertex_faces == -1,
        np.zeros_like(mesh.vertex_faces, dtype=np.float64),
        mesh.area_faces[mesh.vertex_faces],
    )
    areas: npt.NDArray[np.float64] = np.sum(vertex_areas, axis=1) / 3.0
    return areas


def mass_diagonal_mixed_voronoi(mesh: trimesh.Trimesh) -> npt.NDArray[np.float64]:
    """Computes diagonal mass matrix using mixed Voronoi method.

    Uses Voronoi areas for non-obtuse triangles and special weighting for obtuse triangles.

    Args:
        mesh: The input 3D mesh.

    Returns:
        Diagonal entries of the mixed Voronoi mass matrix.
        Shape: (V,) where V is the number of vertices in the mesh.

    Reference:
        Vaillant, R.
        "Compute harmonic weights on a triangular mesh."
        https://mobile.rodolphe-vaillant.fr/entry/20/compute-harmonic-weights-on-a-triangular-mesh
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
    vertex_areas: npt.NDArray[np.float64] = np.zeros_like(mesh.vertices[:, 0])
    np.add.at(vertex_areas, mesh.faces, vertex_triangle_areas)
    return vertex_areas


def fiedler_squared(
    mesh: trimesh.Trimesh, mass_method: str = "mixed_voronoi"
) -> npt.NDArray[np.float64]:
    """Computes squared and normalized Fiedler vector of mesh Laplacian.

    The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
    of the generalized eigenvalue problem Lv = λMv, where L is the cotangent Laplacian
    and M is the mass matrix.

    Args:
        mesh: The input 3D mesh.
        mass_method: Method to compute mass matrix. Either 'barycentric' or 'mixed_voronoi'.

    Returns:
        Squared and normalized Fiedler vector values per vertex.
        Shape: (V,) where V is the number of vertices in the mesh.
    """
    sparse_mass = scipy.sparse.diags(mass_diagonal(mesh, mass_method), format="csr")
    _, v = scipy.sparse.linalg.eigsh(-cotangent_matrix(mesh), M=sparse_mass, k=2, sigma=0)
    field = np.square(v[:, 1])
    scaled_field: npt.NDArray[np.float64] = (field - np.min(field)) / (
        np.max(field) - np.min(field)
    )
    return scaled_field


def gaussian_curvature(mesh: trimesh.Trimesh, eps: float = 1e-14) -> npt.NDArray[np.float64]:
    """Computes Gaussian curvature at each vertex of a triangle mesh.

    This function calculates the Gaussian curvature using the angle defect method, which takes into
    account both interior angles around a vertex and boundary angles for vertices on mesh borders.

    Args:
        mesh: The input 3D mesh.
        eps: Small value to prevent division by zero when normalizing by area.

    Returns:
        Gaussian curvature values per vertex.
        Shape: (V,) where V is the number of vertices in the mesh.

    Note:
        The implementation follows these steps:
        1. Identifies boundary edges and computes angles between them
        2. Subtracts boundary angles from angle defects for boundary vertices
        3. Normalizes by mixed Voronoi areas to get final curvature values

    Reference:
        Vaillant, R.
        "Curvature of a triangle mesh: definition and computation."
        https://rodolphe-vaillant.fr/entry/33/curvature-of-a-triangle-mesh-definition-and-computation
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
    curvature: npt.NDArray[np.float64] = np.divide(
        defects, area_mixed, out=np.zeros_like(area_mixed), where=area_mixed > eps
    )
    return curvature


def mean_curvature(mesh: trimesh.Trimesh, eps: float = 1e-14) -> npt.NDArray[np.float64]:
    """Computes mean curvature at each vertex of a triangle mesh.

    This function calculates the mean curvature using the Laplace-Beltrami operator applied to
    vertex positions, normalized by mixed Voronoi areas.

    Args:
        mesh: The input 3D mesh.
        eps: Small value to prevent division by zero when normalizing by area.

    Returns:
        Mean curvature values per vertex.
        Shape: (V,) where V is the number of vertices in the mesh.

    Note:
        The implementation follows these steps:
        1. Computes cotangent Laplacian matrix
        2. Applies Laplacian to vertex positions
        3. Determines curvature sign using vertex normals
        4. Normalizes by mixed Voronoi areas to get final curvature values

    Reference:
        Vaillant, R.
        "Curvature of a triangle mesh: definition and computation."
        https://rodolphe-vaillant.fr/entry/33/curvature-of-a-triangle-mesh-definition-and-computation
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
    curvature: npt.NDArray[np.float64] = curvature_sign * np.divide(
        unscaled_curvature, 2 * area_mixed, out=np.zeros_like(area_mixed), where=area_mixed > eps
    )
    return curvature


def gframes_lrf(
    mesh: trimesh.Trimesh,
    vertex_index: int,
    radius: float,
    *,
    use_vertex_normal: bool = False,
    scalar_field: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Computes a Local Reference Frame (LRF) for a vertex using the GFrames method.

    This function implements the GFrames (Gradient-based local reference frame) method for computing
    a robust local coordinate system at a given vertex. It uses the gradient of a scalar field
    defined on the mesh surface to determine the x-axis direction.

    Args:
        mesh: The input 3D mesh.
        vertex_index: Index of the vertex for which to compute the LRF.
        radius: Support radius for the LRF computation.
        use_vertex_normal: If True, uses the vertex normal directly as the
            z-axis of the LRF. If False, computes the z-axis from face normals.
        scalar_field: Scalar values defined at each vertex of the mesh.
            Shape: (V,) where V is the number of vertices in the mesh.

    Returns:
        Axes of the LRF stored in columns [x-axis, y-axis, z-axis] forming a right-handed
        coordinate system.
        Shape: (3, 3)

    Note:
        The implementation follows these steps:
        1. Computes z-axis using vertex normal or average of face normals
        2. Selects triangles in the neighborhood based on specified method
        3. Computes first fundamental form coefficients for each triangle
        4. Calculates scalar field gradient using area-weighted contributions
        5. Uses gradient direction for x-axis and completes coordinate system

    Reference:
        Melzi, S., Spezialetti, R., Tombari, F., Bronstein, M. M., Di Stefano, L., & Rodola, E.
        (2019).
        "GFrames: Gradient-based local reference frame for 3D shape matching."
        IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
    """
    if use_vertex_normal:
        z_axis = mesh.vertex_normals[vertex_index]
    else:
        vertex_faces = mesh.vertex_faces[vertex_index]
        vertex_faces = vertex_faces[vertex_faces != -1]
        z_axis = np.mean(mesh.face_normals[vertex_faces], axis=0)
        z_axis = round_zeros(trimesh.transformations.unit_vector(z_axis))

    triangle_indices = get_connected_nearby_face_indices(mesh, vertex_index, radius)
    sqrt_e_coefficients = mesh.edges_unique_length[mesh.faces_unique_edges[triangle_indices, 0]]
    e_coefficients = np.square(sqrt_e_coefficients)
    sqrt_g_coefficients = mesh.edges_unique_length[mesh.faces_unique_edges[triangle_indices, -1]]
    g_coefficients = np.square(sqrt_g_coefficients)
    f_coefficients = (
        sqrt_e_coefficients * sqrt_g_coefficients * np.cos(mesh.face_angles[triangle_indices, 0])
    )
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
    x_axis = round_zeros(
        np.einsum(
            "n,nij,njk,nk->i",
            normalized_triangle_areas,
            edges,
            inverse_matrices,
            scalar_field_differences,
        )
    )
    y_axis = trimesh.transformations.unit_vector(np.cross(z_axis, x_axis))
    x_axis = np.cross(y_axis, z_axis)
    axes: npt.NDArray[np.float64] = np.column_stack((x_axis, y_axis, z_axis))
    return axes


def gframes_frames(
    mesh: trimesh.Trimesh,
    vertex_indices: npt.NDArray[np.int_],
    radius: float,
    *,
    use_vertex_normal: bool = False,
    scalar_field: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Computes Local Reference Frames (LRFs) for multiple vertices using the GFrames method.

    Vectorized version of gframes_lrf that computes LRFs for multiple vertices simultaneously.

    Args:
        mesh: The input 3D mesh.
        vertex_indices: Array of vertex indices for which to compute LRFs.
            Shape: (L,) where L is the number of vertices with LRFs.
        radius: Support radius for the LRF computation.
        use_vertex_normal: If True, uses vertex normals directly as the
            z-axes of the LRFs. If False, computes z-axes from scatter matrix analysis.
        scalar_field: Scalar values defined at each vertex of the mesh.
            Shape: (V,) where V is the number of vertices in the mesh.

    Returns:
        Batch of axes of the LRFs stored in columns [x-axis, y-axis, z-axis] forming
        right-handed coordinate systems.
        Shape: (L, 3, 3)
    """
    vertex_indices = np.atleast_1d(vertex_indices)
    n_vertices = len(vertex_indices)

    if use_vertex_normal:
        z_axes = round_zeros(mesh.vertex_normals[vertex_indices])
    else:
        vertex_faces = mesh.vertex_faces[vertex_indices]
        face_normals = np.vstack((mesh.face_normals, np.zeros(3)))
        z_axes = np.sum(face_normals[vertex_faces], axis=1)
        z_axes = round_zeros(trimesh.transformations.unit_vector(z_axes, axis=-1))

    local_triangle_indices = get_connected_nearby_face_indices(mesh, vertex_indices, radius)
    triangles_counts = np.array(
        [len(triangle_indices) for triangle_indices in local_triangle_indices]
    )
    flat_triangle_indices = np.concatenate(local_triangle_indices)
    frame_indices = np.repeat(np.arange(n_vertices), triangles_counts)
    reduce_indices = np.insert(np.cumsum(triangles_counts)[:-1], 0, 0)

    sqrt_e_coefficients = mesh.edges_unique_length[
        mesh.faces_unique_edges[flat_triangle_indices, 0]
    ]
    e_coefficients = np.square(sqrt_e_coefficients)
    sqrt_g_coefficients = mesh.edges_unique_length[
        mesh.faces_unique_edges[flat_triangle_indices, -1]
    ]
    g_coefficients = np.square(sqrt_g_coefficients)
    f_coefficients = (
        sqrt_e_coefficients
        * sqrt_g_coefficients
        * np.cos(mesh.face_angles[flat_triangle_indices, 0])
    )
    determinants = e_coefficients * g_coefficients - np.square(f_coefficients)
    inverse_matrices = (
        np.array([[g_coefficients, -f_coefficients], [-f_coefficients, e_coefficients]])
        / determinants
    )
    inverse_matrices = np.moveaxis(inverse_matrices, -1, 0)

    triangles = mesh.faces[flat_triangle_indices]
    edges = np.swapaxes(mesh.vertices[triangles[:, 1:]] - mesh.vertices[triangles[:, [0]]], 1, 2)
    scalar_field_differences = np.column_stack(
        [
            scalar_field[triangles[:, 1]] - scalar_field[triangles[:, 0]],
            scalar_field[triangles[:, 2]] - scalar_field[triangles[:, 0]],
        ]
    )
    triangle_areas = mesh.area_faces[flat_triangle_indices]
    triangle_areas_normalizer = np.add.reduceat(triangle_areas, reduce_indices)
    normalized_triangle_areas = triangle_areas / triangle_areas_normalizer[frame_indices]
    x_axes = round_zeros(
        np.add.reduceat(
            np.einsum(
                "n,nij,njk,nk->ni",
                normalized_triangle_areas,
                edges,
                inverse_matrices,
                scalar_field_differences,
            ),
            reduce_indices,
        )
    )

    y_axes = trimesh.transformations.unit_vector(np.cross(z_axes, x_axes), axis=-1)
    x_axes = np.cross(y_axes, z_axes)

    axes: npt.NDArray[np.float64] = np.stack((x_axes, y_axes, z_axes), axis=-1)
    return axes
