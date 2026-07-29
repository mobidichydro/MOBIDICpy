"""Runtime switch for Numba JIT compilation of the numerical kernels.

The kernels decorated with ``@njit``/``@jit`` are called through the function
:func:`dispatch`, which returns either the compiled Numba or the original
Python function (through Numba's ``py_func``). When ``py_func`` is selected,
the code runs as pure Python.

The switch is intended for debugging and for benchmarking the JIT speedup. The code
runs significantly slower as pure Python, so JIT is enabled by default.

The setting is read from ``advanced.jit_enable`` in the YAML configuration and applied
by :class:`~mobidic.core.simulation.Simulation`. Setting ``NUMBA_DISABLE_JIT=1`` in the
environment disables Numba globally at import time and cannot be overridden by the
configuration.
"""

import os

from loguru import logger


def _env_disabled() -> bool:
    """Return True when ``NUMBA_DISABLE_JIT`` is set to a truthy value."""
    return os.environ.get("NUMBA_DISABLE_JIT", "").strip() not in ("", "0")


# When the environment variable NUMBA_DISABLE_JIT is set, the decorators return plain Python functions, so JIT is
# already off before any configuration is read.
_JIT_ENABLED = not _env_disabled()


def jit_enabled() -> bool:
    """Return whether Numba JIT compilation is currently enabled or not."""
    return _JIT_ENABLED


def set_jit_enabled(enabled: bool) -> None:
    """Enable or disable Numba JIT compilation.

    It can be called at any point before or
    between simulations, regardless of import order.

    Args:
        enabled: True to use JIT, False to run them as pure Python.
            A True value is ignored (with a warning) when ``NUMBA_DISABLE_JIT`` is set
            in the environment, since that disables Numba globally.
    """
    global _JIT_ENABLED

    if enabled and _env_disabled():
        logger.warning(
            "jit_enable=true ignored: NUMBA_DISABLE_JIT is set in the environment, "
            "which disables Numba JIT compilation globally."
        )
        _JIT_ENABLED = False
        return

    if enabled != _JIT_ENABLED:
        if enabled:
            logger.info("Numba JIT compilation enabled")
        else:
            logger.warning(
                "Numba JIT compilation disabled: code runs as pure Python, "
                "which is substantially slower. Intended for debugging and benchmarking."
            )

    _JIT_ENABLED = enabled


def dispatch(kernel):
    """Return the callable to use for a Numba-decorated kernel.

    Args:
        kernel: A function decorated with ``@njit``/``@jit``.

    Returns:
        The kernel itself when JIT is enabled, otherwise its pure Python function.

    Examples:
        >>> from mobidic.utils.jit import dispatch
        >>> dispatch(_linear_routing_kernel)(*args)  # doctest: +SKIP
    """
    if _JIT_ENABLED:
        return kernel
    # With NUMBA_DISABLE_JIT set, the decorator already returned a plain function,
    # which has no py_func attribute.
    return getattr(kernel, "py_func", kernel)
