from pathlib import Path

from setuptools import Extension, setup


try:
    import pybind11
except ImportError as exc:
    raise SystemExit("pybind11 is required to build graph_bpe_cpp. Install pybind11 first.") from exc


HERE = Path(__file__).resolve().parent


setup(
    name="graph_bpe_cpp",
    version="0.1.0",
    description="Optional native GraphTokenizer BPE backend.",
    packages=["third_party.graph_bpe_cpp"],
    package_dir={"third_party.graph_bpe_cpp": str(HERE)},
    ext_modules=[
        Extension(
            "third_party.graph_bpe_cpp._graph_bpe",
            sources=[str(HERE / "_graph_bpe.cpp")],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["/std:c++17"] if __import__("os").name == "nt" else ["-std=c++17"],
        )
    ],
)
