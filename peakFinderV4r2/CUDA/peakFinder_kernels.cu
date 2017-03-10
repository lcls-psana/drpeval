#include "peakFinder.h"
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
float *d_data = NULL;
uint *d_conmap = NULL;
uint *d_centers = NULL;
Peak *d_peaks = NULL;
bool *d_stop = NULL;
bool *d_acc_stop = NULL;
__device__ uint* addr_conmap = NULL;
// const uint WIN_TOP = 0;
// const uint WIN_BOT = HEIGHT;
// const uint WIN_LEFT = 0;
// const uint WIN_RIGHT = WIDTH;

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
	cudaEventRecord(t0);
	for(uint imgId = 0; imgId < EVENTS * SHOTS; imgId++)
	{
		filterByThrHigh<<<BLOCKS_PER_SHOT, THREADS_PER_BLOCK>>>(d_data, d_conmap, imgId);
		getCudaError("filterByThrHigh");
	}
	for (int i = 0; i < 2 * rank; i++)
	{
		checkCudaError(cudaMemcpy(d_stop, d_acc_stop, sizeof(bool) * EVENTS * SHOTS, cudaMemcpyDeviceToDevice), "cudaMemcpy, d2d");
		checkCudaError(cudaMemset(d_acc_stop, true, sizeof(bool) * EVENTS * SHOTS), "cudaMemset d_acc_stop");	
		for (uint imgId = 0; imgId < EVENTS * SHOTS; imgId++)
		{
			floodFill<<<BLOCKS_PER_SHOT,THREADS_PER_BLOCK>>>(d_data, d_conmap, d_stop, d_acc_stop, imgId);
			getCudaError("floodFill");
		}
	}
	cudaEventRecord(t1);
	cudaEventSynchronize(t1);
	cudaEventElapsedTime(&miliseconds, t0, t1);
	printf("processing takes %f miliseconds\n", miliseconds);
	if (data_out != NULL)
	{
		checkCudaError(cudaMemcpy(data_out, d_conmap, LSIZE * sizeof(uint), cudaMemcpyDeviceToHost), "cudaMemcpy d2h");
	}

	releaseData();
}