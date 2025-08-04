from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
from pathlib import Path
os.environ["MAX_JOBS"] = "2"

# Resolve paths
here = os.path.dirname(os.path.abspath(__file__))
glm_include_parent = os.path.join(here, "third_party", "glm")  

setup(
    name="gsplat_cuda",
    packages=find_packages(include=["gsplat", "gsplat.*"]),
    ext_modules=[
        CUDAExtension(
            name="gsplat_cuda._C",
            sources=[
                "gsplat/cuda/csrc/compute_3D_smoothing_filter_fwd.cu",
                "gsplat/cuda/csrc/compute_relocation.cu",
                "gsplat/cuda/csrc/compute_sh_fwd.cu",
                "gsplat/cuda/csrc/ext.cpp",
                "gsplat/cuda/csrc/fully_fused_projection_fwd.cu",
                "gsplat/cuda/csrc/fully_fused_projection_packed_fwd.cu",
                "gsplat/cuda/csrc/isect_tiles.cu",
                "gsplat/cuda/csrc/quat_scale_to_covar_preci_fwd.cu",
                "gsplat/cuda/csrc/raytracing_to_pixels_fwd.cu",
                "gsplat/cuda/csrc/view_to_gaussians_fwd.cu",
                "gsplat/cuda/csrc/world_to_cam_fwd.cu",
            ],
            include_dirs=[glm_include_parent],
            extra_compile_args={
                "nvcc":
                ["-Xcompiler",
                 "-fno-gnu-unique",
                 "--expt-relaxed-constexpr",
                 "-O3",
                 "--use_fast_math",
                ]}
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
