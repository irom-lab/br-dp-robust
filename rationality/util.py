import matplotlib.pyplot as plt
import os

import jax
import jax.numpy as jnp

from typing import Iterable, Optional
from rationality.types import State

def in_ipynb():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def ordinal(n: int) -> str:
    return "%d%s" % (n, 'tsnrhtdd' [(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])


def savefig(base_name: str, formats: Iterable[str], pad=0.0) -> None:
    for format in formats:
        filename = f'{base_name}.{format}'
        print(f'Saving {format.upper()} format: {filename}')

        if os.path.exists(filename):
            os.remove(filename)

        if format in {'pgf'}:
            plt.savefig(f'{base_name}.{format}', format=format, bbox_inches='tight', pad_inches=pad, backend='pgf')
        elif format in {'eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff'}:
            plt.savefig(f'{base_name}.{format}', format=format, bbox_inches='tight', pad_inches=pad)
        else:
            raise ValueError(f"Unknown file format: `{format}'.")


@jax.jit
def transform(x: jnp.ndarray, rot_mat: jnp.ndarray, centroid: jnp.ndarray) -> jnp.ndarray:
    return rot_mat @ x + centroid


def draw_quad(ax: plt.Axes, state: State, color='k', body_size=10, arm_length: float = 1.0,
              prop_height: float = 1.0, prop_width: float = 1.0):
    conf = state[:3]
    centroid = conf[:2].reshape((-1, 1))
    angle = conf[3]

    rot_mat = jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                         [jnp.sin(angle), jnp.cos(angle)]])

    arm_endpoints = jnp.array([[-arm_length, arm_length], [0.0, 0.0]])
    prop_centers = arm_endpoints + jnp.array([[0.0, 0.0], [prop_height, prop_height]])
    left_prop = jnp.stack([[-prop_width / 2.0, prop_width / 2.0], [0.0, 0.0]]) + prop_centers[:, 0].reshape((-1, 1))
    right_prop = jnp.stack([[-prop_width / 2.0, prop_width / 2.0], [0.0, 0.0]]) + prop_centers[:, 1].reshape((-1, 1))

    arm_endpoints = transform(arm_endpoints, rot_mat, centroid)
    prop_centers = transform(prop_centers, rot_mat, centroid)
    left_prop = transform(left_prop, rot_mat, centroid)
    right_prop = transform(right_prop, rot_mat, centroid)

    ax.scatter(centroid[0], centroid[1], s=body_size, c=color)
    ax.plot(arm_endpoints[0, :], arm_endpoints[1, :], color=color)
    ax.plot((arm_endpoints[0, 0], prop_centers[0, 0]), (arm_endpoints[1, 0], prop_centers[1, 0]), color=color)
    ax.plot((arm_endpoints[0, 1], prop_centers[0, 1]), (arm_endpoints[1, 1], prop_centers[1, 1]), color=color)

    ax.plot(left_prop[0, :], left_prop[1, :], color=color)
    ax.plot(right_prop[0, :], right_prop[1, :], color=color)


