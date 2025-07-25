#ifndef CUDA_CULLER_FORWARD_H_INCLUDED
#define CUDA_CULLER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* view2gaussian_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const float kernel_size,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* view2gaussians,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered, float* area);
	

	void gCull(
		const dim3 tile_bounds, dim3 block,
		const int width, int height,
		const bool* bool_mask,
		const float focal_x, float focal_y,
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		const float *__restrict__ view2gaussian,
		const float3 *__restrict__ scales,
		const float4 *__restrict__ conic_opacity,
		bool* output
	);
	
	glm::mat3 quat2rot(const glm::vec4 q);
}
#endif