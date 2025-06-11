/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void test(int* y);
		
		static void skycull(
			const dim3 tile_bounds, dim3 block,
			const int width, int height,
			const bool* sky,
			const float focal_x, float focal_y,
			const uint2 *__restrict__ ranges,
			const uint32_t *__restrict__ point_list,
			const float *__restrict__ view2gaussian,
			const int *__restrict__ gaussian_index,
			const float3 *__restrict__ scales,
			const float4 *__restrict__ conic_opacity,
			bool* output);	

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const bool* bool_mask,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const int* gaussian_index,
			const float* cov3D_precomp,
			const float* view2gaussian_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const float kernel_size,
			const float* subpixel_offset,
			const bool prefiltered,
			bool* output,
			int* radii = nullptr,
			bool debug = false);
	};
};
#endif