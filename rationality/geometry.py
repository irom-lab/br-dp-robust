import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from typing import Tuple, Optional, NamedTuple, Callable


class AxisAlignedBoundingBox(NamedTuple):
    centroid: jnp.ndarray
    dimensions: jnp.ndarray

    @jax.jit
    def contains(self, point: jnp.ndarray) -> bool:
        centered = point - self.centroid

        return jnp.all((centered <= (self.dimensions / 2) + 1e-6) & (-self.dimensions / 2 <= (centered + 1e-6)))

    @jax.jit
    def project(self, point: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        def case_point_not_in_box():
            centered = point - self.centroid
            proj = jnp.maximum(jnp.minimum(centered, self.dimensions / 2), -self.dimensions / 2) + self.centroid

            return proj, jnp.linalg.norm(proj - point, ord=2)

        return jax.lax.cond(self.contains(point),
                            lambda _: (point, 0.0),
                            lambda _: case_point_not_in_box(),
                            None)

    @jax.jit
    def intersects(self, start: jnp.ndarray, end: jnp.ndarray) -> bool:
        @jax.jit
        def dimension_scanner(carry: None, dimension_information: jnp.ndarray) -> tuple[None, bool]:
            s, e, c, d = dimension_information

            A = jnp.stack([jnp.array([s, e]) - c, jnp.ones(2)], axis=-1)

            sln_lower = jnp.linalg.lstsq(A, jnp.array([[-d / 2], [1.0]]), rcond=None)[0].flatten()
            sln_upper = jnp.linalg.lstsq(A, jnp.array([[d / 2], [1.0]]), rcond=None)[0].flatten()

            sln_lower = sln_lower / sln_lower.sum()
            sln_upper = sln_upper / sln_upper.sum()

            intersection_found = (sln_lower > 0) and self.contains(jnp.stack([start, end], axis=-1) @ sln_lower)
            intersection_found &= (sln_upper > 0) and self.contains(jnp.stack([start, end], axis=-1) @ sln_upper)

            return None, intersection_found

        dimensional_info = (start, end, self.centroid, self.dimensions)

        return self.contains(start) or self.contains(end) or \
               jnp.any(jax.lax.scan(dimension_scanner, None, dimensional_info)[1])


class Workspace(NamedTuple):
    boundary: AxisAlignedBoundingBox
    obstacles: AxisAlignedBoundingBox

    @jax.jit
    def freespace_contains_point(self, point: jnp.ndarray) -> bool:
        return self.boundary.contains(point) and not jnp.any(jax.vmap(jax.jit(lambda c, d: aabb(c, d).contains(point)))(self.obstacles.centroid, self.obstacles.dimensions))

    @jax.jit
    def freespace_contains_segment(self, start: jnp.ndarray, end: jnp.ndarray) -> bool:
        return self.boundary.contains(start) and self.boundary.contains(end) and not\
               jax.vmap(jax.jit(lambda c, d: aabb(c, d).intersects(start, end)))(self.obstacles.centroid, self.obstacles.dimensions)


@jax.jit
def aabb(centroid: jnp.ndarray, dimensions: jnp.ndarray) -> AxisAlignedBoundingBox:
    return AxisAlignedBoundingBox(centroid, dimensions)


def workspace(width: float, height: float, obstacles: list[AxisAlignedBoundingBox]):
    obstacles = aabb(jnp.stack([o.centroid for o in obstacles]), jnp.stack([o.dimensions for o in obstacles]))

    return Workspace(aabb(jnp.array([width / 2.0, height / 2.0]), jnp.array([width, height])), obstacles)


def draw(aabb: AxisAlignedBoundingBox, ax: plt.Axes, hatch: Optional[str] = None) -> plt.Axes:
    centroid, dimensions = aabb

    verts = jnp.array([[dimensions[0], -dimensions[1]],
                       [-dimensions[0], -dimensions[1]],
                       [-dimensions[0], dimensions[1]],
                       [dimensions[0], dimensions[1]]]) / 2 + centroid.reshape((1, -1))

    if hatch is not None:
        ax.add_patch(plt.Polygon(verts, fill=False, edgecolor='k', hatch=hatch))
    else:
        ax.add_patch(plt.Polygon(verts, color='k'))

    return ax
