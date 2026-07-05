import platform

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


extra_cuda_cflags = ["-O2"]
if platform.system() == "Windows":
    extra_cuda_cflags.append("--use-local-env")

setup(
    name="tetris-env-cuda",
    version="0.1.0",
    description="CUDA vectorized Tetris environment for PyTorch",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="tetris_env._C",
            sources=["src/TetrisEnv.cu"],
            extra_compile_args={
                "cxx": ["-g"],
                "nvcc": extra_cuda_cflags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "ninja",
    ],
    include_package_data=True,
    zip_safe=False,
)
