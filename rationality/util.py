import matplotlib.pyplot as plt
import os

from typing import Iterable, Optional

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


def savefig(base_name: str, formats: Iterable[str]) -> None:
    for format in formats:
        filename = f'{base_name}.{format}'
        print(f'Saving {format.upper()} format: {filename}')

        if os.path.exists(filename):
            os.remove(filename)

        if format in {'pgf'}:
            plt.savefig(f'{base_name}.{format}', format=format, bbox_inches='tight', pad_inches=0.0, backend='pgf')
        elif format in {'eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff'}:
            plt.savefig(f'{base_name}.{format}', format=format, bbox_inches='tight', pad_inches=0.0)
        else:
            raise ValueError(f"Unknown file format: `{format}'.")
