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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
							 const float *orig_points,
							 const float *viewmatrix,
							 const float *projmatrix,
							 bool *present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2 *points_xy,
	const float *depths,
	const uint32_t *offsets,
	uint64_t *gaussian_keys_unsorted,
	uint32_t *gaussian_values_unsorted,
	int *radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth.
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t *)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

__global__ void createWithKeys(
	int P,
	const float2 *points_xy,
	const float *depths,
	const uint32_t *offsets,
	const uint32_t *tiles_touched,
	uint64_t *points_keys_unsorted,
	uint32_t *points_values_unsorted,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Points
	if (tiles_touched[idx] > 0)
	{
		// Find this Point's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

		// determine the tile that the point is in
		const float2 p = points_xy[idx];
		int x = min(grid.x - 1, max((int)0, (int)(p.x / BLOCK_X)));
		int y = min(grid.y - 1, max((int)0, (int)(p.y / BLOCK_Y)));

		uint64_t key = y * grid.x + x;
		key <<= 32;
		key |= *((uint32_t *)&depths[idx]);
		points_keys_unsorted[off] = key;
		points_values_unsorted[off] = idx;
	}
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t *point_list_keys, uint2 *ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char *&chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.view2gaussian, P * 16, 128); // could be 12
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::PointState CudaRasterizer::PointState::fromChunk(char *&chunk, size_t P)
{
	PointState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.points2D, P, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char *&chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N * 4, 128);
	obtain(chunk, img.center_depth, N, 128);
	obtain(chunk, img.center_alphas, N, 128);
	obtain(chunk, img.n_contrib, N * 2, 128);
	obtain(chunk, img.ranges, N, 128);
	obtain(chunk, img.point_ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char *&chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

void CudaRasterizer::Rasterizer::test(int *y)
{
	FORWARD::test(y);
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char *(size_t)> geometryBuffer,
	std::function<char *(size_t)> binningBuffer,
	std::function<char *(size_t)> imageBuffer,
	const int P, int D, int M,
	const float *background,
	const int width, int height,
	const float *means3D,
	const float *shs,
	const float *colors_precomp,
	const float *opacities,
	const float *scales,
	const float scale_modifier,
	const float *rotations,
	const int *gaussian_index,
	const float *cov3D_precomp,
	const float *view2gaussian_precomp,
	const float *viewmatrix,
	const float *projmatrix,
	const float *cam_pos,
	const float tan_fovx, float tan_fovy,
	const float kernel_size,
	const float *subpixel_offset,
	const bool prefiltered,
	bool *output,
	int *radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char *chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char *img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Allocate and intialize host array
	float *hArea = (float *)malloc(P * sizeof(float));
	memset(hArea, 0, P * sizeof(float));

	// Allocate device array
	float *dArea;
	cudaMalloc(&dArea, P * sizeof(float));
	cudaMemcpy(dArea, hArea, P * sizeof(float), cudaMemcpyHostToDevice);

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
				   P, D, M,
				   means3D,
				   (glm::vec3 *)scales,
				   scale_modifier,
				   (glm::vec4 *)rotations,
				   opacities,
				   shs,
				   geomState.clamped,
				   cov3D_precomp,
				   colors_precomp,
				   view2gaussian_precomp,
				   viewmatrix, projmatrix,
				   (glm::vec3 *)cam_pos,
				   width, height,
				   focal_x, focal_y,
				   tan_fovx, tan_fovy,
				   kernel_size,
				   radii,
				   geomState.means2D,
				   geomState.depths,
				   geomState.cov3D,
				   geomState.view2gaussian,
				   geomState.rgb,
				   geomState.conic_opacity,
				   tile_grid,
				   geomState.tiles_touched,
				   prefiltered, dArea),
			   debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char *binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys<<<(P + 255) / 256, 256>>>(
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
		CHECK_CUDA(, debug)

			int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
				   binningState.list_sorting_space,
				   binningState.sorting_size,
				   binningState.point_list_keys_unsorted, binningState.point_list_keys,
				   binningState.point_list_unsorted, binningState.point_list,
				   num_rendered, 0, 32 + bit),
			   debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// printf("in CudaRasterizer::Rasterizer::forward, P: %d num_rendered: %d geo_chunk_size: %d img_chunk_size: %d, binning_chunk_size: %d\n", P, num_rendered, chunk_size, img_chunk_size, binning_chunk_size);
	//  Let each tile blend its range of Gaussians independently in parallel
	const float *feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	// const float* cov3Ds = cov3D_precomp != nullptr ? cov3D_precomp : geomState.cov3D;
	const float *view2gaussian = view2gaussian_precomp != nullptr ? view2gaussian_precomp : geomState.view2gaussian;
	// const float* view2gaussian = view2gaussian_precomp;

	bool* sky_mask = NULL;
	CHECK_CUDA(FORWARD::skycull(
		tile_grid, block,
		width, height,
		sky_mask,
		focal_x, focal_y,
		imgState.ranges,
		binningState.point_list,
		view2gaussian,
		(int *)gaussian_index,
		(float3 *)scales,
		geomState.conic_opacity,
		output
	), debug);

	return num_rendered;
}