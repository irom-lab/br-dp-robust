import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from typing import Tuple, Optional, NamedTuple, Callable


class AxisAlignedBoundingBox(NamedTuple):
    centroid: jnp.ndarray
    dimensions: jnp.ndarray

    contains: Callable[[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], bool]
    project: Callable[[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], Tuple[jnp.ndarray, float]]
    intersects: Callable[[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], bool]


class Workspace(NamedTuple):
    boundary: AxisAlignedBoundingBox
    obstacles: list[AxisAlignedBoundingBox]


def aabb(centroid: jnp.ndarray, dimensions: jnp.ndarray) -> AxisAlignedBoundingBox:
    return AxisAlignedBoundingBox(centroid, dimensions, _aabb_contains, _aabb_project, _aabb_intersects)


def workspace(width: float, height: float, obstacles: list[AxisAlignedBoundingBox]):
    return Workspace(aabb(jnp.ndarray([width / 2, height / 2]), jnp.ndarray([width / 2, height / 2])), obstacles)


@jax.jit
def _aabb_contains(aabb_params: tuple[jnp.ndarray, jnp.ndarray], point: jnp.ndarray) -> bool:
    centroid, dimensions = aabb_params
    centered = point - centroid

    return jnp.all((centered <= (dimensions / 2) + 1e-6) & (-dimensions / 2 <= (centered + 1e-6)))


@jax.jit
def _aabb_project(aabb_params: tuple[jnp.ndarray, jnp.ndarray], point: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
    centroid, dimensions = aabb_params

    def case_point_not_in_box():
        centered = point - centroid
        proj = jnp.maximum(jnp.minimum(centered, dimensions / 2), -dimensions / 2) + centroid

        return proj, jnp.linalg.norm(proj - point, ord=2)

    return jax.lax.cond(_aabb_contains(aabb_params, point),
                        lambda _: (point, 0.0),
                        lambda _: case_point_not_in_box(),
                        None)


@jax.jit
def _aabb_intersects(aabb_params: tuple[jnp.ndarray, jnp.ndarray], start: jnp.ndarray, end: jnp.ndarray) -> bool:
    centroid, dimensions = aabb_params

    @jax.jit
    def dimension_scanner(carry: None, dimension_information: jnp.ndarray) -> tuple[None, bool]:
        s, e, c, d = dimension_information

        A = jnp.stack([jnp.array([s, e]) - c, jnp.ones(2)], axis=-1)

        sln_lower = jnp.linalg.lstsq(A, jnp.array([[-d / 2], [1.0]]), rcond=None)[0].flatten()
        sln_upper = jnp.linalg.lstsq(A, jnp.array([[d / 2], [1.0]]), rcond=None)[0].flatten()

        sln_lower = sln_lower / sln_lower.sum()
        sln_upper = sln_upper / sln_upper.sum()

        intersection_found = (sln_lower > 0) and _aabb_contains(aabb_params,
                                                                jnp.stack([start, end], axis=-1) @ sln_lower)
        intersection_found &= (sln_upper > 0) and _aabb_contains(aabb_params,
                                                                 jnp.stack([start, end], axis=-1) @ sln_upper)

        return None, intersection_found

    dimensional_info = (start, end, centroid, dimensions)

    return _aabb_contains(start) or _aabb_contains(end) or \
           jnp.any(jax.lax.scan(dimension_scanner, None, dimensional_info)[1])


def draw(aabb: AxisAlignedBoundingBox, ax: plt.Axes, hatch: Optional[str] = None) -> plt.Axes:
    centroid, dimensions, _, _, _ = aabb

    verts = jnp.array([[dimensions[0], -dimensions[1]],
                       [-dimensions[0], -dimensions[1]],
                       [-dimensions[0], dimensions[1]],
                       [dimensions[0], dimensions[1]]]) / 2 + centroid.reshape((1, -1))

    if hatch is not None:
        ax.add_patch(plt.Polygon(verts, fill=False, edgecolor='k', hatch=hatch))
    else:
        ax.add_patch(plt.Polygon(verts, color='k'))

    return ax
