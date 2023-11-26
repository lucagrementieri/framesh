import functools
import time
from typing import Optional

import numpy as np
import numpy.typing as npt
import trimesh
import trimesh.util


def timeit(method):
    @functools.wraps(method)
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


@timeit
def rops_local_reference_frame(
    mesh: trimesh.Trimesh,
    center: npt.NDArray[np.float64],
    radius: Optional[float] = None,
):
    differences = mesh.vertices - center
    distances = trimesh.util.row_norm(differences)
    if radius is None:
        radius = np.max(distances)
    d1 = np.expand_dims(differences[mesh.faces], axis=(2, 4))
    d2 = np.expand_dims(differences[mesh.faces], axis=(1, 3))
    dw = np.expand_dims(np.eye(3) + 1, (0, 3, 4))
    face_scatter = np.sum(dw * d1 * d2, axis=(1, 2)) / 12
    area_weights = mesh.area_faces / mesh.area
    face_centers = np.mean(mesh.vertices[mesh.faces], axis=1)
    centers_differences = face_centers - center
    distance_weights = np.square(radius - trimesh.util.row_norm(centers_differences))
    mesh_scatter = np.sum(
        face_scatter * np.expand_dims(area_weights * distance_weights, axis=(1, 2)),
        axis=0,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(mesh_scatter)
    assert eigenvalues[0] <= eigenvalues[1]
    assert eigenvalues[1] <= eigenvalues[2]
    axes = np.fliplr(eigenvectors)
    hx = np.sum(centers_differences.dot(axes[0]) * area_weights * distance_weights)
    if hx < 0:
        axes[0] *= -1
    hz = np.sum(centers_differences.dot(axes[2]) * area_weights * distance_weights)
    if hz < 0:
        axes[2] *= -1
    axes[1] = np.cross(axes[2], axes[0])
    return axes


if __name__ == "__main__":
    from pathlib import Path

    import trimesh

    patch_path = Path(__file__).parents[1] / "patches" / "patch_000.ply"
    patch_mesh = trimesh.load_mesh(patch_path)
    centroid = patch_mesh.centroid
    print(rops_local_reference_frame(patch_mesh, centroid))
