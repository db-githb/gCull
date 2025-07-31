#ifndef CUDA_CULLER_H_INCLUDED
#define CUDA_CULLER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaCuller
{
	class Culler
	{
	public:
		
		static void gCull(
			const dim3 tile_bounds, dim3 block,
			const int width, int height,
			const bool* sky,
			const float focal_x, float focal_y,
			const uint2 *__restrict__ ranges,
			const uint32_t *__restrict__ point_list,
			const float *__restrict__ view2gaussian,
			const float3 *__restrict__ scales,
			const float4 *__restrict__ conic_opacity,
			int* output);	

		static void forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const int* binary_mask,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* view2gaussian_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const float kernel_size,
			const float* subpixel_offset,
			const bool prefiltered,
			int* output,
			int* radii = nullptr,
			bool debug = false);
	};
};
#endif