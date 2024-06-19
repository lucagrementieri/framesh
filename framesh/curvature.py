from typing import Tuple

import igl
import numpy as np
import numpy.typing as npt
import trimesh


def gaussian_mean_curvatures(
    vertices: npt.NDArray[np.floating],
    triangles: npt.NDArray[np.integer],
    vertex_normals: npt.NDArray[np.floating],
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    tmesh = trimesh.Trimesh(vertices, triangles)
    n = vertices.shape[0]
    area_mixed = np.zeros(n)
    e01v = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    e20v = vertices[triangles[:, 0]] - vertices[triangles[:, 2]]
    e12v = vertices[triangles[:, 2]] - vertices[triangles[:, 1]]
    angles = tmesh.face_angles

    gaussian_curvature = np.copy(tmesh.vertex_defects)

    cotan_angles = 1.0 / np.tan(angles)

    area_mixed = igl.massmatrix(vertices, triangles).diagonal()
    total_curvature = np.zeros_like(vertices)
    np.add.at(
        total_curvature,
        triangles[:, 0],
        (e20v * cotan_angles[:, [1]] - e01v * cotan_angles[:, [2]]) / 4.0,
    )
    np.add.at(
        total_curvature,
        triangles[:, 1],
        (e01v * cotan_angles[:, [2]] - e12v * cotan_angles[:, [0]]) / 4.0,
    )
    np.add.at(
        total_curvature,
        triangles[:, 2],
        (e12v * cotan_angles[:, [0]] - e20v * cotan_angles[:, [1]]) / 4.0,
    )

    wrapped_triangles = np.pad(triangles, ((0, 0), (0, 1)), mode="wrap")
    edges = np.lib.stride_tricks.sliding_window_view(wrapped_triangles, 2, axis=1)
    edges_list = np.reshape(np.sort(edges, axis=-1), (-1, 2))
    _, edge_inverse, edge_counts = np.unique(
        edges_list, return_inverse=True, return_counts=True, axis=0
    )
    border_indices = np.flatnonzero(edge_counts == 1)
    is_border = np.reshape(np.isin(edge_inverse, border_indices), (-1, 3))

    border_edges = edges[is_border]
    double_border_edges = np.row_stack([border_edges, np.fliplr(border_edges)])
    order = np.lexsort(np.rot90(double_border_edges))
    original_mask = order < border_edges.shape[0]
    double_border_edges = double_border_edges[order]
    start = double_border_edges[::2, 0]
    end1 = double_border_edges[::2, 1]
    end2 = double_border_edges[1::2, 1]
    e1 = vertices[end1] - vertices[start]
    e2 = vertices[end2] - vertices[start]

    consistent_border_edges = np.logical_xor(original_mask[::2], original_mask[1::2])
    e2[np.logical_not(consistent_border_edges)] *= -1
    gaussian_curvature[start] -= _angle(e1, e2)

    non_null_area = area_mixed > np.finfo(np.float64).eps
    gaussian_curvature = np.divide(
        gaussian_curvature,
        area_mixed,
        out=np.zeros_like(gaussian_curvature),
        where=non_null_area,
    )
    # sign seems a much better choice instead of where
    mean_curvature_sign = np.where(np.sum(total_curvature * vertex_normals, axis=-1) > 0, 1.0, -1.0)
    mean_curvature = mean_curvature_sign * np.divide(
        np.linalg.norm(total_curvature, axis=-1),
        area_mixed,
        out=np.zeros_like(gaussian_curvature),
        where=non_null_area,
    )
    return gaussian_curvature, mean_curvature


def _angle(u: npt.NDArray[np.floating], v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    u_norm = np.linalg.norm(u, axis=-1)
    v_norm = np.linalg.norm(v, axis=-1)
    norm_product = u_norm * v_norm
    cos_angle: npt.NDArray[np.floating] = np.divide(
        np.sum(u * v, axis=-1),
        norm_product,
        out=np.ones_like(norm_product),
        where=norm_product > np.finfo(np.float64).eps,
    )
    cos_angle = np.clip(cos_angle, -1, 1)
    arccos: npt.NDArray[np.floating] = np.arccos(cos_angle)
    return arccos
