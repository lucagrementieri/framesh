import functools
import time

import numpy as np
import numpy.typing as npt
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
def shot_local_reference_frame(
    points: npt.NDArray[np.float64], center: npt.NDArray[np.float64]
):
    difference = points - center
    distances = trimesh.util.row_norm(difference)
    radius = np.max(distances)
    scale_factors = radius - distances
    scale_factors /= scale_factors.sum()
    weighted_covariance = np.einsum(
        "i,ij,ik->jk", scale_factors, difference, difference
    )
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_covariance)
    assert eigenvalues[0] <= eigenvalues[1]
    assert eigenvalues[1] <= eigenvalues[2]
    axes = np.fliplr(eigenvectors)
    if np.mean(np.dot(difference, axes[0]) >= 0) < 0.5:
        axes[0] *= -1
    if np.mean(np.dot(difference, axes[2]) >= 0) < 0.5:
        axes[2] *= -1
    if np.dot(np.cross(axes[2], axes[0]), axes[1]) < 0:
        axes[1] *= -1
    return axes


if __name__ == "__main__":
    from pathlib import Path

    import trimesh

    patch_path = Path(__file__).parents[1] / "patches" / "patch_000.ply"
    patch_mesh = trimesh.load_mesh(patch_path)
    vertices = patch_mesh.vertices
    centroid = patch_mesh.centroid
    print(shot_local_reference_frame(vertices, centroid))
