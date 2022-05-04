import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from typing import Tuple, Optional, NamedTuple, Callable


class Polytope(NamedTuple):
    linear: jnp.ndarray
    affine: jnp.ndarray
    centroid: jnp.ndarray
    dimensions: jnp.ndarray

    @jax.jit
    def contains(self, point: jnp.ndarray) -> bool:

        return jnp.all(self.linear @ point <= self.affine + 1e-6)

    @jax.jit
    def intersects(self, start: jnp.ndarray, end: jnp.ndarray) -> bool:
        @jax.jit
        def halfspace_scanner(carry: bool, face_info: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[bool, bool]:
            a, b = face_info

            l = (b - (a @ end)) / (a @ (start - end))
            pt = l * start + (1.0 - l) * end

            found_intersection = (((0.0 <= l) & (l <= 1.0)) & self.contains(pt))

            return carry | found_intersection, found_intersection

        return self.contains(start) | self.contains(end) | jax.lax.scan(halfspace_scanner, False, (self.linear, self.affine))[0]


class Workspace(NamedTuple):
    boundary: Polytope
    obstacles: Polytope

@jax.jit
def freespace_contains_point(w: Workspace, point: jnp.ndarray) -> bool:
    return w.boundary.contains(point) & ~jnp.any(jax.vmap(jax.jit(lambda A, b, c, d: Polytope(A, b, c, d).contains(point)))(*w.obstacles))

@jax.jit
def freespace_contains_segment(w: Workspace, start: jnp.ndarray, end: jnp.ndarray) -> bool:
    return w.boundary.contains(start) & w.boundary.contains(end) &\
            ~jnp.any(jax.vmap(jax.jit(lambda A, b, c, d: Polytope(A, b, c, d).intersects(start, end)))(*w.obstacles))


def aabb(centroid: jnp.ndarray, dimensions: jnp.ndarray) -> Polytope:
    n = len(dimensions)

    normals = jnp.array([jnp.zeros(n).at[i].set(1.0) for i in range(n)] + [jnp.zeros(n).at[i].set(-1.0) for i in range(n)])
    affine = jnp.array([normals[i, :] @ (0.5 * dimensions[i] * normals[i, :] + centroid) for i in range(n)] +
                       [normals[n + i, :] @ (0.5 * dimensions[i] * normals[n + i, :] + centroid) for i in range(n)])

    return Polytope(normals, affine, centroid, dimensions)


def workspace(width: float, height: float, obstacles: list[Polytope]) -> Workspace:
    obstacles = Polytope(jnp.stack([o.linear for o in obstacles]), jnp.stack([o.affine for o in obstacles]),
                         jnp.stack([o.centroid for o in obstacles]), jnp.stack([o.dimensions for o in obstacles]))

    return Workspace(aabb(jnp.array([width / 2.0, height / 2.0]), jnp.array([width, height])), obstacles)


def draw(aabb: Polytope, ax: plt.Axes, hatch: Optional[str] = None) -> plt.Axes:
    _, _, centroid, dimensions = aabb

    verts = jnp.array([[dimensions[0], -dimensions[1]],
                       [-dimensions[0], -dimensions[1]],
                       [-dimensions[0], dimensions[1]],
                       [dimensions[0], dimensions[1]]]) / 2 + centroid.reshape((1, -1))

    if hatch is not None:
        ax.add_patch(plt.Polygon(verts, fill=False, edgecolor='k'))
    else:
        ax.add_patch(plt.Polygon(verts, color='k'))

    return ax
