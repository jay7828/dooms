"""Builds the dooms wheel from a staged directory made by scripts/package-python.sh (the Python sources plus dooms/lib with libdooms_c.dylib and its
dependencies, already relocated to @loader_path). Not meant to be run by hand."""
import os

from setuptools import Distribution, find_packages, setup
from wheel.bdist_wheel import bdist_wheel

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "VERSION")) as f:
    version = f.read().strip()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):  # the package carries compiled libraries: make it a platform wheel
        return True



class PlatformWheel(bdist_wheel):
    def get_tag(self):  # pure Python code over a ctypes library: any Python 3 on Apple Silicon macOS, no CPython ABI involved
        return "py3", "none", "macosx_11_0_arm64"


setup(
    name="dooms",
    version=version,
    description="dooms.cpp: LLM text generation (any GGUF), image question answering (SmolVLM) and robot actions (SmolVLA) on Apple Silicon",
    packages=find_packages(include=["dooms", "dooms.*"]),
    package_data={"dooms": ["lib/*.dylib"]},
    include_package_data=True,
    install_requires=["numpy"],
    python_requires=">=3.9",
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": PlatformWheel},
    classifiers=["Operating System :: MacOS :: MacOS X", "Programming Language :: Python :: 3", "Topic :: Scientific/Engineering :: Artificial Intelligence"],
)
