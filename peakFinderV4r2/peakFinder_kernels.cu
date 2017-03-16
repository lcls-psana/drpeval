#include "peakFinder.h"
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
float *d_data = NULL;
uint *d_conmap = NULL;
uint *d_centers = NULL;
uint *d_dense_centers = NULL;
Peak *d_peaks = NULL;
bool *d_stop = NULL;
bool *d_acc_stop = NULL;
__device__ uint* addr_conmap = NULL;

const int THREADS_PER_BLOCK = 256;
const int BLOCKS_PER_SHOT = (WIDTH * HEIGHT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

__global__ void filterByThrHigh(const float *d_data, uint* d_conmap, uint imgId)
{
	uint pixId = (blockIdx.x % BLOCKS_PER_SHOT) * blockDim.x + threadIdx.x;
	// imgId = blockIdx.x / BLOCKS_PER_SHOT;
	if (pixId < WIDTH * HEIGHT)
	{
		uint dataId = imgId * WIDTH * HEIGHT + pixId;
		if (d_data[dataId] > thr_high)
			d_conmap[dataId] = dataId;
	}
}

__global__ void floodFill(const float *d_data, uint* d_conmap, bool *d_stop, bool *d_acc_stop, uint imgId)
{
	// imgId = blockIdx.x / BLOCKS_PER_SHOT;
	if (d_stop[imgId]) return;
	uint pixId = (blockIdx.x % BLOCKS_PER_SHOT) * blockDim.x + threadIdx.x;
	if (pixId < WIDTH * HEIGHT)
	{
		uint dataId = imgId * WIDTH * HEIGHT + pixId;
		float intensity = d_data[dataId];
		if (intensity > thr_low)
		{
			uint row = pixId / WIDTH, col = pixId % WIDTH;
			uint status = d_conmap[dataId];
			bool changed = false;
			if (col > 0)
			{
				uint tgt_Id = d_conmap[dataId-1];
				if (tgt_Id > 0 && (status == 0 || d_data[tgt_Id] > intensity) && col - tgt_Id % WIDTH <= rank)
				{
					d_conmap[dataId] = tgt_Id; changed = true;
				}
			}
			if (col < WIDTH - 1)
			{
				uint tgt_Id = d_conmap[dataId+1];
				if (tgt_Id > 0 && (status == 0 || d_data[imgId * WIDTH * HEIGHT + tgt_Id] > intensity) && tgt_Id % WIDTH - col <= rank)
				{
					d_conmap[dataId] = tgt_Id; changed = true;
				}
			}	
			if (row > 0)
			{
				uint tgt_Id = d_conmap[dataId-WIDTH];
				if (tgt_Id > 0 && (status == 0 || d_data[imgId * WIDTH * HEIGHT + tgt_Id] > intensity) && row - (tgt_Id / WIDTH) % HEIGHT <= rank)
				{
					d_conmap[dataId] = tgt_Id; changed = true;
				}
			}
			if (row < HEIGHT - 1)
			{
				uint tgt_Id = d_conmap[dataId+WIDTH];
				if (tgt_Id > 0 && (status == 0 || d_data[imgId * WIDTH * HEIGHT + tgt_Id] > intensity) && (tgt_Id / WIDTH) % HEIGHT - row <= rank)
				{
					d_conmap[dataId] = tgt_Id; changed = true;
				}
			}
			if (changed) d_acc_stop[imgId] = false;
		}
	}
}

__global__ void floodFill_sanity(const float *d_data, uint* d_conmap, uint imgId)
{
	// imgId = blockIdx.x / BLOCKS_PER_SHOT;
	// uint imgId = blockIdx.x * blockDim.x + threadIdx.x;
	uint pixId = (blockIdx.x % BLOCKS_PER_SHOT) * blockDim.x + threadIdx.x;
	if (pixId < WIDTH * HEIGHT && imgId < EVENTS * SHOTS)
	{
		uint dataId = imgId * WIDTH * HEIGHT + pixId;
		float intensity = d_data[dataId];
		if (intensity > thr_low)
		{
			uint row = pixId / WIDTH, col = pixId % WIDTH;
			uint status = d_conmap[dataId];
			if (col > 0)
			{
				uint tgt_Id = d_conmap[dataId-1];
				if (tgt_Id > 0 && (status == 0 || d_data[tgt_Id] > intensity) && col - tgt_Id % WIDTH <= rank)
				{
					d_conmap[dataId] = tgt_Id;
				}
			}
			if (col < WIDTH - 1)
			{
				uint tgt_Id = d_conmap[dataId+1];
				if (tgt_Id > 0 && (status == 0 || d_data[tgt_Id] > intensity) && tgt_Id % WIDTH - col <= rank)
				{
					d_conmap[dataId] = tgt_Id;
				}
			}	
			if (row > 0)
			{
				uint tgt_Id = d_conmap[dataId-WIDTH];
				if (tgt_Id > 0 && (status == 0 || d_data[tgt_Id] > intensity) && row - (tgt_Id / WIDTH) % HEIGHT <= rank)
				{
					d_conmap[dataId] = tgt_Id;
				}
			}
			if (row < HEIGHT - 1)
			{
				uint tgt_Id = d_conmap[dataId+WIDTH];
				if (tgt_Id > 0 && (status == 0 || d_data[tgt_Id] > intensity) && (tgt_Id / WIDTH) % HEIGHT - row <= rank)
				{
					d_conmap[dataId] = tgt_Id;
				}
			}
		}
	}
}

// const int PATCH_WIDTH = 2 * rank + 1;
// const int THREADS_PER_PATCH = 128;
// const int PIX_PER_PATCH = (2 * rank + 1) * (2 * rank + 1);
// const int PATCH_PER_BLOCK = 1;
// const int THREADS_PER_BLOCK_v2 = PATCH_PER_BLOCK * THREADS_PER_PATCH;

const int FILTER_PATCH_WIDTH = 32;
const int FILTER_PATCH_HEIGHT = 8; 
const int FILTER_THREADS_PER_PATCH = FILTER_PATCH_WIDTH * FILTER_PATCH_HEIGHT;
const int FILTER_PATCH_ON_WIDTH = (WIDTH) / FILTER_PATCH_WIDTH;
const int FILTER_PATCH_ON_HEIGHT = (HEIGHT + FILTER_PATCH_HEIGHT - 1) / FILTER_PATCH_HEIGHT;
const int FILTER_PATCH_PER_IMAGE = FILTER_PATCH_ON_WIDTH * FILTER_PATCH_ON_HEIGHT;
__global__ void filterByThrHigh_v2(const float *d_data, uint *d_centers)
{
	uint imgId = blockIdx.x / FILTER_PATCH_PER_IMAGE;
	uint patch_id = blockIdx.x % FILTER_PATCH_PER_IMAGE;
	uint patch_x = patch_id % FILTER_PATCH_ON_WIDTH;
	uint patch_y = patch_id / FILTER_PATCH_ON_WIDTH;
	__shared__ float data[FILTER_PATCH_HEIGHT * FILTER_PATCH_WIDTH];
	__shared__ uint idxs[FILTER_PATCH_HEIGHT * FILTER_PATCH_WIDTH];
	int irow = threadIdx.x / FILTER_PATCH_WIDTH;
	int icol = threadIdx.x % FILTER_PATCH_WIDTH;
	int row = patch_y * FILTER_PATCH_HEIGHT + irow;
	int col = patch_x * FILTER_PATCH_WIDTH + icol;
	const int NUM_NMS_AREA = FILTER_PATCH_WIDTH / FILTER_PATCH_HEIGHT;
	int local_area = icol / FILTER_PATCH_HEIGHT;
	int local_pos = local_area * (FILTER_PATCH_HEIGHT * FILTER_PATCH_HEIGHT) + irow * FILTER_PATCH_HEIGHT + icol % FILTER_PATCH_HEIGHT;
	uint device_pos = imgId * (WIDTH * HEIGHT) + row * WIDTH + col;
	__shared__ bool has_candidate[NUM_NMS_AREA];
	if (threadIdx.x < NUM_NMS_AREA) has_candidate[threadIdx.x] = false;
	__syncthreads();
	// load data
	if (row < WIDTH && col < HEIGHT){
		data[local_pos] = d_data[device_pos];
		idxs[local_pos] = device_pos;
	}
	else{
		data[local_pos] = 0;
	}

	if (data[local_pos] > thr_high)
		has_candidate[local_area] = true;
	__syncthreads();
	// find maximum
	local_area = threadIdx.x / (FILTER_PATCH_HEIGHT * FILTER_PATCH_HEIGHT);
	if (!has_candidate[local_area])
		return;
	const int local_tid = threadIdx.x % (FILTER_PATCH_HEIGHT * FILTER_PATCH_HEIGHT);
	const int local_offset =  local_area * (FILTER_PATCH_HEIGHT * FILTER_PATCH_HEIGHT);
	int num_of_working_threads = (FILTER_PATCH_HEIGHT * FILTER_PATCH_HEIGHT) / 2;
	if (local_tid >= num_of_working_threads) return;
	int idx_mul = 1;
	while (num_of_working_threads > 1 && local_tid < num_of_working_threads)
	{
		int idx1 = (local_tid * 2) * idx_mul + local_offset;
		int idx2 = idx1 + idx_mul;
		int idxm = data[idx1] > data[idx2] ? idx1 : idx2;
		data[idx1] = data[idxm];
		idxs[idx1] = idxs[idxm];
		__syncthreads();
		idx_mul *= 2;
		num_of_working_threads /= 2;
	}
	if (local_tid < NUM_NMS_AREA)
	{
		uint write_pos = blockIdx.x * NUM_NMS_AREA + local_area;
		d_centers[write_pos] = idxs[local_offset];
	}
}

// if (imgId == 9){
// 	uint dp = idxs[local_offset];
// 	int dr = (dp / WIDTH) % HEIGHT;
// 	int dc = (dp % WIDTH);
// 	printf("imgId:%d, local_tid:%d, local_area:%d, threadIdx.x:%d, row:%d, col:%d\n", imgId, local_tid, local_area, threadIdx.x, dr, dc);
// }
const int PATCH_WIDTH = (2 * rank + 1);
const int FF_LOAD_THREADS_PER_CENTER = 64;
const int FF_THREADS_PER_CENTER = 32;
// const int FF_SIDE_WIDTH = FF_THREADS_PER_CENTER / 4;
// const int FF_SIDE_OFFSET = 1 - FF_SIDE_WIDTH / 2;
const int FF_THREADS_PER_BLOCK = 64;
const int FF_LOAD_PASS = (2 * rank + 1) * (2 * rank + 1) / FF_LOAD_THREADS_PER_CENTER + 1;
const int FF_CENTERS_PER_BLOCK = FF_THREADS_PER_BLOCK / FF_LOAD_THREADS_PER_CENTER;
// one center per block
__global__ void floodFill_v2(const float *d_data, const uint *d_centers, uint *d_conmap)
{
	const uint center_id = d_centers[blockIdx.x];
	const uint img_id = center_id / (WIDTH * HEIGHT);
	const uint crow = center_id / WIDTH % HEIGHT;
	const uint ccol = center_id % WIDTH;
	__shared__ float data[PATCH_WIDTH][PATCH_WIDTH];
	__shared__ uint status[PATCH_WIDTH][PATCH_WIDTH];
	// load data
	for (int i = 0; i < FF_LOAD_PASS; i++)
	{
		const uint tmp_id = i * FF_LOAD_THREADS_PER_CENTER + threadIdx.x;
		const uint irow = tmp_id / PATCH_WIDTH;
		const uint icol = tmp_id % PATCH_WIDTH;
		const int drow = crow + irow - rank;
		const int dcol = ccol + icol - rank;
		if (drow >= 0 && drow < HEIGHT && dcol >= 0 && dcol < WIDTH)
		{
			data[irow][icol] = d_data[img_id * (WIDTH * HEIGHT) + drow * WIDTH + dcol];
		}
		else if(irow < PATCH_WIDTH)
		{
			data[irow][icol] = 0;
		}
	}
	for(int i = 0; i < FF_LOAD_PASS; i++)
	{
		const uint tmp_id = i * FF_LOAD_THREADS_PER_CENTER + threadIdx.x;
		const uint irow = tmp_id / PATCH_WIDTH;
		const uint icol = tmp_id % PATCH_WIDTH;
		if (irow < PATCH_WIDTH){
			status[irow][icol] = 0;
		}
		if (irow == rank && icol == rank){
			status[irow][icol] = 1;
		}
	}
	__syncthreads();
	if (threadIdx.x >= FF_THREADS_PER_CENTER)
		return;
	// flood fill
	const int FF_SCAN_LENGTH = FF_THREADS_PER_CENTER / 8;
	const int sign_x[8] = {-1, 1, 1, -1, 1, 1, -1, -1};
	const int sign_y[8] = {1, 1, -1, -1, 1, -1, -1, 1};
	const int scanline_id = threadIdx.x / FF_SCAN_LENGTH;
	const int base_v = threadIdx.x % (2 * FF_SCAN_LENGTH) - FF_SCAN_LENGTH;
	int icol = base_v * sign_x[scanline_id] + rank;
	int irow = base_v * sign_y[scanline_id] + rank;
	const int scangrp_id = threadIdx.x / (2 * FF_SCAN_LENGTH);
	const int dxs[4] = {-1, 1, 0, 0};
	const int dys[4] = {0, 0, 1, -1};
	const int dx = dxs[scangrp_id];
	const int dy = dys[scangrp_id];
	// __shared__ bool is_local_maximum;
	// is_local_maximum = true;
	for(int i = 1; i <= rank; i++){
		__syncthreads();
		icol += dx;
		irow += dy;
		if (data[irow][icol] > thr_low){
			if (status[irow-dy][icol-dx] > 0){
				status[irow][icol] = center_id;
			}
		}
		// if (i == 3){
		// 	status[irow][icol] = 100;
		// }
	}
	// write data
	const int FF_WRITE_PASS = (PATCH_WIDTH * PATCH_WIDTH + FF_THREADS_PER_CENTER - 1) / FF_THREADS_PER_CENTER;
	for(int i = 0; i < FF_WRITE_PASS; i++){
		const uint tmp_id = i * FF_THREADS_PER_CENTER + threadIdx.x;
		const uint irow = tmp_id / PATCH_WIDTH;
		const uint icol = tmp_id % PATCH_WIDTH;
		const int drow = crow + irow - rank;
		const int dcol = ccol + icol - rank;
		if (irow < PATCH_WIDTH && status[irow][icol] > 0 && drow >= 0 && drow < HEIGHT && dcol >= 0 && dcol < WIDTH)
		{
			// if(img_id == 9)
			// 	printf("irow:%d, icol:%d, center_id:%d\n", irow, icol, center_id);
			d_conmap[img_id * (WIDTH * HEIGHT) + drow * WIDTH + dcol] = status[irow][icol];
		}
	}

}

struct is_center
{
	__device__
	bool operator()(const uint &x){
		// return x == addr_conmap[x];
		return x > 0;
	}
};

__global__ void setConmap(uint* d_conmap)
{
	addr_conmap = d_conmap;
}

void checkCudaError(cudaError_t err, const char* msg)
{
	if (err != cudaSuccess)
	{
		printf("failed: %s\n, error code: %s\n", msg, cudaGetErrorString(err));
	}
}

void getCudaError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("failed: %s\n, error code %s\n", msg, cudaGetErrorString(err));
	}
}

void setUpData(float *h_data)
{
	checkCudaError(cudaMalloc((void **)&d_data, LSIZE * sizeof(float)), "cudaMalloc d_data");
	checkCudaError(cudaMalloc((void **)&d_conmap, LSIZE * sizeof(uint)), "cudaMalloc d_conmap");
	checkCudaError(cudaMalloc((void **)&d_stop, EVENTS * SHOTS * sizeof(bool)), "cudaMalloc d_stop");
	checkCudaError(cudaMalloc((void **)&d_acc_stop, EVENTS * SHOTS * sizeof(bool)), "cudaMalloc d_acc_stop");
	checkCudaError(cudaMemset(d_conmap, 0, sizeof(uint)*LSIZE), "cudaMemset d_conmap");
	checkCudaError(cudaMemset(d_acc_stop, false, sizeof(bool) * EVENTS * SHOTS), "cudaMemset d_acc_stop");
	checkCudaError(cudaMemcpy(d_data, h_data, LSIZE * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy h2d");
}

void releaseData()
{
	cudaFree(d_data);
	cudaFree(d_conmap);
}


extern "C" void processImages(float *data, Peak *peak_out, uint *data_out)
{
	float miliseconds = 0.0f;
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);
	cudaEventRecord(t0);
	setUpData(data);
	cudaEventRecord(t1);
	cudaEventSynchronize(t1);
	cudaEventElapsedTime(&miliseconds, t0, t1);
	printf("passing data to gpu takes %f miliseconds\n", miliseconds);
	// cudaEventRecord(t0);
	// floodFill v1
	// for(uint imgId = 0; imgId < EVENTS * SHOTS; imgId++)
	// {
	// 	filterByThrHigh<<<BLOCKS_PER_SHOT, THREADS_PER_BLOCK>>>(d_data, d_conmap, imgId);
	// 	getCudaError("filterByThrHigh");
	// }
	// for (int i = 0; i < 2 * rank; i++)
	// {
	// 	checkCudaError(cudaMemcpy(d_stop, d_acc_stop, sizeof(bool) * EVENTS * SHOTS, cudaMemcpyDeviceToDevice), "cudaMemcpy, d2d");
	// 	checkCudaError(cudaMemset(d_acc_stop, true, sizeof(bool) * EVENTS * SHOTS), "cudaMemset d_acc_stop");	
	// 	for (uint imgId = 0; imgId < EVENTS * SHOTS; imgId++)
	// 	{
	// 		floodFill<<<BLOCKS_PER_SHOT,THREADS_PER_BLOCK>>>(d_data, d_conmap, d_stop, d_acc_stop, imgId);
	// 		getCudaError("floodFill");
	// 	}
	// }
	// sanity check small
	// for (int i = 0; i < 2 * rank; i++){
	// 	floodFill_sanity<<<(LSIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_data, d_conmap);
	// 	getCudaError("floodfill sanity");
	// }

	// Flood Fill sanity
	// for (int i = 0; i < 2 * rank; i++)
	// {
	// 	for (uint imgId = 0; imgId < EVENTS * SHOTS; imgId++)
	// 	{
	// 		floodFill_sanity<<<BLOCKS_PER_SHOT, THREADS_PER_BLOCK>>>(d_data, d_conmap, imgId);
	// 		getCudaError("floodfill sanity");
	// 	}
	// }

	// floodFill v2
	printf("filterByThrHigh_v2: num_blocks:%ld\n", FILTER_PATCH_PER_IMAGE * EVENTS * SHOTS);
	const int centers_size = FILTER_PATCH_PER_IMAGE * (FILTER_PATCH_WIDTH / FILTER_PATCH_HEIGHT) * EVENTS * SHOTS;
	checkCudaError(cudaMalloc((void **)&d_centers, centers_size * sizeof(uint)), "cudaMalloc d_centers");
	checkCudaError(cudaMemset(d_centers, 0, centers_size * sizeof(uint)), "cudaMemset d_centers");
	checkCudaError(cudaMalloc((void **)&d_dense_centers, centers_size * sizeof(uint)), "cudaMalloc d_dense_centers");
	cudaDeviceSynchronize();
	cudaEventRecord(t0);
	filterByThrHigh_v2<<<FILTER_PATCH_PER_IMAGE * EVENTS * SHOTS, FILTER_THREADS_PER_PATCH>>>(d_data, d_centers);
	getCudaError("filterByThrHigh_v2");
	cudaEventRecord(t1);
	cudaEventSynchronize(t1);
	cudaEventElapsedTime(&miliseconds, t0, t1);
	printf("filterByThrHigh_v2 takes %f miliseconds\n", miliseconds);
	cudaEventRecord(t0);
	thrust::device_ptr<uint> dp_dense_centers = thrust::device_pointer_cast(d_dense_centers);
	thrust::device_ptr<uint> dp_centers = thrust::device_pointer_cast(d_centers);
	auto end_centers = thrust::copy_if(dp_centers, dp_centers + centers_size,  dp_dense_centers, is_center());
	int num_pix = end_centers - dp_dense_centers;
	printf("num of testing pixels:%d\n", num_pix);
	cudaEventRecord(t1);
	cudaEventSynchronize(t1);
	cudaEventElapsedTime(&miliseconds, t0, t1);
	printf("stream compaction takes %f miliseconds\n", miliseconds);

	const int NUM_BLOCKS = num_pix / FF_CENTERS_PER_BLOCK;

	cudaEventRecord(t0);
	floodFill_v2<<<NUM_BLOCKS, FF_THREADS_PER_BLOCK>>>(d_data, d_dense_centers, d_conmap);
	cudaEventRecord(t1);
	cudaEventSynchronize(t1);
	cudaEventElapsedTime(&miliseconds, t0, t1);
	printf("floodFill_v2 takes %f miliseconds\n", miliseconds);
	getCudaError("floodFill_v2");

	if (data_out != NULL)
	{
		checkCudaError(cudaMemcpy(data_out, d_conmap, LSIZE * sizeof(uint), cudaMemcpyDeviceToHost), "cudaMemcpy d2h");
	}

	releaseData();
}