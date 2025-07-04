#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

#from scene.gaussian_model import GaussianModel
#from utils.sh_utils import eval_sh

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def cull_gaussians(
    bool_mask,
    means3D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    view2gaussian_precomp,
    raster_settings,
):
    return _CullGaussians.apply(
        bool_mask,
        means3D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        view2gaussian_precomp,
        raster_settings,
    )

class _CullGaussians(torch.autograd.Function):

    grad_offsets = torch.tensor([])

    @staticmethod
    def forward(
        ctx,
        bool_mask,
        means3D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        view2gaussian_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            bool_mask,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            view2gaussian_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.kernel_size,
            raster_settings.subpixel_offset,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA culler
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, cull_list, radii, geomBuffer, binningBuffer, imgBuffer = _C.cull_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, cull_list, radii, geomBuffer, binningBuffer, imgBuffer = _C.cull_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, view2gaussian_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return cull_list

class GaussianCullSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    kernel_size : float
    subpixel_offset: torch.Tensor
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianCuller(nn.Module):
    def __init__(self, cull_settings):
        super().__init__()
        self.cull_settings = cull_settings

    def forward(self, bool_mask, means3D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, view2gaussian_precomp = None):
        
        cull_settings = self.cull_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # TODO check and raise exception for precomputed view2gaussian
        if view2gaussian_precomp is None:
            view2gaussian_precomp = torch.Tensor([])
            
        # Invoke C++/CUDA rasterization routine
        return cull_gaussians(
            bool_mask,
            means3D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            view2gaussian_precomp,
            cull_settings, 
        )
