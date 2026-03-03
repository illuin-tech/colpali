from importlib.metadata import PackageNotFoundError, version

from .models import *

try:
    __version__ = version("colpali_engine")
except PackageNotFoundError:
    __version__ = "0.0.0"
