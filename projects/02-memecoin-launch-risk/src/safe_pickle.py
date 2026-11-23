"""Open a pickle file without letting it run code.

Why: a pickle file can contain instructions that run on your computer the moment
it is opened. feature.pkl comes from a Google Drive link, so we open it through
this loader. It accepts only plain containers (dict, list, tuple, set, OrderedDict),
numbers, text, and numpy arrays. Anything else is refused BEFORE it runs.

What it does not do: it does not make a hostile file harmless in every way (for
example, a file can still ask for a huge array and use up memory). It stops code
from running. Still only open files from the official source, and check the hash.
"""

import pickle

# module -> names that may be rebuilt. Everything not listed is refused.
_ALLOWED = {
    "collections": {"OrderedDict"},
    "builtins": {"set", "frozenset", "bytearray"},
    "_codecs": {"encode"},  # how older pickles store bytes inside numpy arrays
    "numpy": {"ndarray", "dtype"},
    # numpy renamed core -> _core in version 2; files made by either are accepted
    "numpy.core.multiarray": {"_reconstruct", "scalar"},
    "numpy._core.multiarray": {"_reconstruct", "scalar"},
    "numpy.core.numeric": {"_frombuffer"},
    "numpy._core.numeric": {"_frombuffer"},
}


class UnsafePickleError(pickle.UnpicklingError):
    """The file asked for something that is not on the allowed list."""


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in _ALLOWED.get(module, ()):
            return super().find_class(module, name)
        raise UnsafePickleError(f"refused to load {module}.{name}: not on the allowed list")


def load(file_object):
    """Load a pickle from an open binary file, refusing anything not allowed."""
    return _RestrictedUnpickler(file_object).load()


def loads(data: bytes):
    """Load a pickle from bytes, refusing anything not allowed."""
    import io

    return load(io.BytesIO(data))


def load_file(path):
    """Load a pickle from a file path, refusing anything not allowed."""
    with open(path, "rb") as f:
        return load(f)
