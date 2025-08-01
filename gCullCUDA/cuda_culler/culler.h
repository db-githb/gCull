#ifndef CUDA_CULLER_H_INCLUDED
#define CUDA_CULLER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaCuller
{
	class Culler
	{
	public:	

		static void forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
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
			float* out_color,
			int* radii = nullptr,
			bool debug = false);
	};
};
#endif