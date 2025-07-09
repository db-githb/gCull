from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="gCullCUDA",
    packages=["gCullCUDA"],
    ext_modules=[
        CUDAExtension(
            name="gCullCUDA._C",
            sources=[
            "cuda_culler/culler_impl.cu",
            "cuda_culler/forward.cu",
            "cull_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
