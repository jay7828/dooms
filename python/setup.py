"""Builds the dooms wheel from a staged directory made by scripts/package-python.sh (the Python sources plus dooms/lib with libdooms_c.dylib and its
dependencies, already relocated to @loader_path, README.md, VERSION, the licence files). Not meant to be run by hand: the script also computes the platform tag
from the libraries it bundles and passes it in DOOMS_WHEEL_PLATFORM."""
import os

from setuptools import Distribution, find_packages, setup
from wheel.bdist_wheel import bdist_wheel

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "VERSION")) as f:
    version = f.read().strip()
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

DISTRIBUTION_NAME = "dooms-cpp"   # the PyPI name ("dooms" is taken there by an unrelated project); the import name stays `dooms`
REPOSITORY = "https://github.com/jay7828/dooms"
PLATFORM_ENV = "DOOMS_WHEEL_PLATFORM"


class BinaryDistribution(Distribution):
    def has_ext_modules(self):  # the package carries compiled libraries: make it a platform wheel
        return True


class PlatformWheel(bdist_wheel):
    def get_tag(self):  # pure Python code over a ctypes library: any Python 3 on Apple Silicon macOS, no CPython ABI involved
        plat = os.environ.get(PLATFORM_ENV, "")
        if not (plat.startswith("macosx_") and plat.endswith("_arm64")):
            raise SystemExit(f"{PLATFORM_ENV} must name the macOS arm64 platform the bundled libraries need (e.g. macosx_14_0_arm64); "
                             "build the wheel with scripts/package-python.sh, which sets it")
        return "py3", "none", plat


setup(
    name=DISTRIBUTION_NAME,
    version=version,
    description="dooms.cpp on Apple Silicon: LLM text generation (any GGUF) with an OpenAI-compatible server, image question answering (SmolVLM) "
                "and robot actions (SmolVLA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=REPOSITORY,
    project_urls={"Source": REPOSITORY, "Documentation": REPOSITORY + "/blob/main/docs/USING.md", "Issues": REPOSITORY + "/issues"},
    license="Apache-2.0",
    platforms=["macOS arm64"],
    packages=find_packages(include=["dooms", "dooms.*"]),
    package_data={"dooms": ["lib/*.dylib"]},
    include_package_data=True,
    install_requires=["numpy"],
    python_requires=">=3.9",
    entry_points={"console_scripts": ["dooms=dooms.cli:main"]},  # `dooms serve | pull | list`; also `python -m dooms`
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": PlatformWheel},
    keywords=["inference", "llm", "gguf", "apple-silicon", "metal", "coreml", "openai-compatible"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Environment :: GPU",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
