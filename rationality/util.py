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
