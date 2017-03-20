#include "peakFinder.h"
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <functional>
float *d_data = NULL;
uint *d_conmap = NULL;
uint *d_centers = NULL;
uint *d_dense_centers = NULL;
Peak *d_peaks = NULL;

const int FILTER_PATCH_WIDTH = 32;
const int FILTER_PATCH_HEIGHT = 4; 
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
	// if (local_tid >= num_of_working_threads) return;
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
	if (local_tid == 0)
	{
		uint write_pos = blockIdx.x * NUM_NMS_AREA + local_area;
		d_centers[write_pos] = idxs[local_offset];
	}
}

const int PATCH_WIDTH = (2 * HALF_WIDTH + 1);
const int FF_LOAD_THREADS_PER_CENTER = 64;
const int FF_THREADS_PER_CENTER = 32;
const int FF_INFO_THREADS_PER_CENTER = FF_THREADS_PER_CENTER;
// const int FF_SIDE_WIDTH = FF_THREADS_PER_CENTER / 4;
// const int FF_SIDE_OFFSET = 1 - FF_SIDE_WIDTH / 2;
const int FF_THREADS_PER_BLOCK = 64;
const int FF_LOAD_PASS = (2 * HALF_WIDTH + 1) * (2 * HALF_WIDTH + 1) / FF_LOAD_THREADS_PER_CENTER + 1;
const int FF_CENTERS_PER_BLOCK = FF_THREADS_PER_BLOCK / FF_LOAD_THREADS_PER_CENTER;

__device__ void calPreSum(int *preSum)
{
	for(int i = 1; i < FF_INFO_THREADS_PER_CENTER; i*=2){
		int prevIdx = threadIdx.x - i;
		int sum = preSum[threadIdx.x];
		if (prevIdx > 0){
			sum += preSum[prevIdx];
		}
		__syncthreads();
		preSum[threadIdx.x] = sum;
		__syncthreads();
	}
}

typedef float (*reducer) (const float &, const float &); 
typedef float (*transformer) (const int &);

__device__ float deviceAdd(const float &a, const float &b) {return a+b;}
__device__ float deviceMin(const float &a, const float &b) {return a<b?a:b;}
__device__ float deviceMax(const float &a, const float &b) {return a>b?a:b;}

const int WARP_SIZE = 32;
__device__ __inline__ float warpReduce(float val, int npix, reducer r)
{
	int offset = 32;
	if (npix < 32)
	{
		if (npix > 16) offset = 16;
		else if (npix > 8) offset = 8;
		else if (npix > 4) offset = 4;
		else if (npix > 2) offset = 2;
		else if (npix > 1) offset = 1;
		else offset = 0;
	}
	for(; offset > 0; offset /= 2){
		int srcIdx = threadIdx.x + offset;
		float nVal = __shfl_down(val, offset);
		if (srcIdx < npix){
			val = r(val, nVal);
		}
	}
	return val;
}
__device__ __inline__ float blockReduce(int npix, transformer t, reducer r)
{
	const int FF_PIX_PASS = (npix + FF_INFO_THREADS_PER_CENTER - 1) / FF_INFO_THREADS_PER_CENTER;
	__shared__ float buffer[32];
	for(int i = 0; i < FF_PIX_PASS; i++){
		uint tmp_id = i * FF_INFO_THREADS_PER_CENTER + threadIdx.x;
		int n = WARP_SIZE;
		if (i == FF_PIX_PASS - 1){
			n = npix % WARP_SIZE;
		}
		float val = warpReduce(t(tmp_id), n, r);
		if (threadIdx.x == 0){
			buffer[i] = val;
		}
	}
	return warpReduce(buffer[threadIdx.x], FF_PIX_PASS, r);
}

__device__ __inline__ bool inRing(int dr, int dc)
{
	float dist2 = dr * dr + dc * dc;
	const float lower = r0 * r0;
	const float upper = (r0 + dr) * (r0 + dr);
	return dist2 >= lower && dist2 <= upper;
}

__device__ __inline__ bool peakIsPreSelected(float son, float npix, float amp_max, float amp_tot)
{
	if (son < peak_son_min) return false;
	if (npix < peak_npix_min) return false;
	if (npix > peak_npix_max) return false;
	if (amp_max < peak_amax_thr) return false;
	if (amp_tot < peak_atot_thr) return false;
	return true; 
}
// one center per block
__global__ void floodFill_v2(const float *d_data, const uint *d_centers, Peak *d_peaks, uint *d_conmap)
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
		const int drow = crow + irow - HALF_WIDTH;
		const int dcol = ccol + icol - HALF_WIDTH;
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
		if (irow == HALF_WIDTH && icol == HALF_WIDTH){
			status[irow][icol] = center_id;
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
	const int id_in_grp = threadIdx.x % (2 * FF_SCAN_LENGTH);
	const int base_v = id_in_grp - FF_SCAN_LENGTH;
	int icol = base_v * sign_x[scanline_id] + HALF_WIDTH;
	int irow = base_v * sign_y[scanline_id] + HALF_WIDTH;
	const int scangrp_id = threadIdx.x / (2 * FF_SCAN_LENGTH);
	const int dxs[4] = {-1, 1, 0, 0};
	const int dys[4] = {0, 0, 1, -1};
	const int dx = dxs[scangrp_id];
	const int dy = dys[scangrp_id];
	const float center_intensity = data[HALF_WIDTH][HALF_WIDTH];
	__shared__ bool is_local_maximum;
	is_local_maximum = true;
	for(int i = 1; i <= rank; i++){
		__syncthreads();
		if (!is_local_maximum) return;
		icol += dx;
		irow += dy;
		if (data[irow][icol] > center_intensity){
			is_local_maximum = false;
		}
		if (data[irow][icol] > thr_low){
			if (status[irow-dy][icol-dx] == center_id){
				status[irow][icol] = center_id;
			}
		}
	}

	const int bound = base_v > 0 ? base_v : -base_v;
	for(int i = 1; i <= FF_SCAN_LENGTH - 1; i++){
		__syncthreads();
		if (!is_local_maximum) return;
		if (i > bound) continue;
		icol += dx;
		irow += dy;
		// if (irow >= PATCH_WIDTH || icol >= PATCH_WIDTH || irow < 0 || icol < 0){
		// 	printf("irow:%d, icol:%d, img_id:%d, i:%d\n", irow, icol, img_id, i);
		// }
		if (data[irow][icol] > center_intensity){
			is_local_maximum = false;
		}
		if (data[irow][icol] > thr_low){
			if (status[irow-dy][icol-dx] == center_id){
				status[irow][icol] = center_id;
			}
		}
	} 

	const int FF_PROC_PASS = (PATCH_WIDTH * PATCH_WIDTH + FF_INFO_THREADS_PER_CENTER - 1) / FF_INFO_THREADS_PER_CENTER;
	// calculate peak info
	__shared__ float peak_data[PATCH_WIDTH * PATCH_WIDTH];
	__shared__ int peak_row[PATCH_WIDTH * PATCH_WIDTH];
	__shared__ int peak_col[PATCH_WIDTH * PATCH_WIDTH];
	// data compaction
	__shared__ int preSum[FF_INFO_THREADS_PER_CENTER];
	preSum[threadIdx.x] = 0;
	for(int i = 0; i < FF_PROC_PASS; i++){
		const uint tmp_id = i * FF_INFO_THREADS_PER_CENTER + threadIdx.x;
		const uint irow = tmp_id / PATCH_WIDTH;
		const uint icol = tmp_id % PATCH_WIDTH;
		if (irow < PATCH_WIDTH && status[irow][icol] == center_id){
			preSum[threadIdx.x] += 1;
		}
	}
	calPreSum(preSum);
	int npix = preSum[FF_INFO_THREADS_PER_CENTER - 1];
	int counter = 0;
	__shared__ float bg_avg;
	__shared__ float bg_rms;
	__shared__ float bg_npix;
	if (threadIdx.x == 0){
		bg_avg = 0;
		bg_rms = 0;
		bg_npix = 0;
	}
	for(int i = 0; i < FF_PROC_PASS; i++){
		const uint tmp_id = i * FF_INFO_THREADS_PER_CENTER + threadIdx.x;
		const uint irow = tmp_id / PATCH_WIDTH;
		const uint icol = tmp_id % PATCH_WIDTH;
		if (irow < PATCH_WIDTH){
			if (status[irow][icol] == center_id){
				int pos = counter;
				if (threadIdx.x > 0)
					pos += preSum[threadIdx.x - 1];
				peak_data[pos] = data[irow][icol];
				peak_row[pos] = irow;
				peak_col[pos] = icol;
				counter ++;
			}
			// calculate background info
			if (inRing(irow-HALF_WIDTH,icol-HALF_WIDTH) && data[irow][icol] < thr_low){
				float d = data[irow][icol];
				atomicAdd(&bg_avg,d);
				atomicAdd(&bg_rms,d * d);
				atomicAdd(&bg_npix, 1);
			}
		}
	}
	const int FF_PIX_PASS = (npix + FF_INFO_THREADS_PER_CENTER - 1) / FF_INFO_THREADS_PER_CENTER;
	__shared__ float buffer[32];
	#define BLOCK_REDUCE(v,t,r) \
	for(int i = 0; i < FF_PIX_PASS; i++){ \
		uint tmp_id = i * FF_INFO_THREADS_PER_CENTER + threadIdx.x; \
		int n = WARP_SIZE; \
		if (i == FF_PIX_PASS - 1){ \
			n = npix % WARP_SIZE; \
		} \
		float val = warpReduce(t(tmp_id), n, r); \
		if (threadIdx.x == 0){ \
			buffer[i] = val; \
		} \
	} \
	v= warpReduce(buffer[threadIdx.x], FF_PIX_PASS, r); 
	float samp; BLOCK_REDUCE(samp, [=]__device__(const int &id) -> float {return peak_data[id];}, deviceAdd);
	__shared__ Peak peak;
	if (threadIdx.x == 0)
	{
		bg_avg /= bg_npix;
		bg_rms = bg_rms / bg_npix - bg_avg * bg_avg;
		bg_rms = sqrtf(bg_rms);
		float noise_tot = bg_rms * sqrtf(npix);
		peak.amp_tot = samp - bg_avg * npix;
		peak.amp_max = center_intensity - bg_avg;
		peak.son = noise_tot > 0 ? peak.amp_tot / noise_tot : 0;
		peak.bkgd = bg_avg;
		peak.noise = bg_rms;
		peak.valid = peakIsPreSelected(peak.son, npix, peak.amp_max, peak.amp_tot);
	}
	__syncthreads();
	if (!peak.valid) return;
	// if (center_id == 4734401 && threadIdx.x == 0)
	// 	printf("sum of intensity:%f,\n", samp);
	float rmin; BLOCK_REDUCE(rmin, [=]__device__(const int &id) -> float {return peak_row[id];}, deviceMin);
	float rmax; BLOCK_REDUCE(rmax, [=]__device__(const int &id) -> float {return peak_row[id];}, deviceMax);
	float cmin; BLOCK_REDUCE(cmin, [=]__device__(const int &id) -> float {return peak_col[id];}, deviceMin);
	float cmax; BLOCK_REDUCE(cmax, [=]__device__(const int &id) -> float {return peak_col[id];}, deviceMax);
	float sar1; BLOCK_REDUCE(sar1, [=]__device__(const int &id) -> float {return peak_data[id] * peak_row[id];}, deviceAdd);
	float sac1; BLOCK_REDUCE(sac1, [=]__device__(const int &id) -> float {return peak_data[id] * peak_col[id];}, deviceAdd);
	float sar2; BLOCK_REDUCE(sar2, [=]__device__(const int &id) -> float {return peak_data[id] * peak_row[id] * peak_row[id];}, deviceAdd);
	float sac2; BLOCK_REDUCE(sac2, [=]__device__(const int &id) -> float {return peak_data[id] * peak_col[id] * peak_col[id];}, deviceAdd);
	if(threadIdx.x == 0){
		peak.evt = img_id / SHOTS;
		peak.seg = img_id % SHOTS;
		// printf("center_id: %d, evt: %f, seg:%f\n", center_id, peak.evt, peak.seg);
		peak.row = crow;
		peak.col = ccol;
		peak.npix = npix;
		peak.row_min = rmin;
		peak.row_max = rmax;
		peak.col_min = cmin;
		peak.col_max = cmax;
		if (samp > 0){
			sar1 = sar1 / samp;
			sac1 = sac1 / samp;
			sar2 = sar2 / samp - sar1 * sar1;
			sac2 = sac2 / samp - sac1 * sac1;
			peak.row_cgrav = sar1;
			peak.col_cgrav = sac1;
			peak.row_sigma = (npix > 1 && sar2 > 0) ? sqrtf(sar2) : 0;
			peak.col_sigma = (npix > 1 && sac2 > 0) ? sqrtf(sac2) : 0;
		}
		else {
			peak.row_cgrav = crow;
			peak.col_cgrav = ccol;
			peak.row_sigma = 0;
			peak.col_sigma = 0;
		}
		d_peaks[blockIdx.x] = peak;
	}

	// printf("center_id:%d\n", center_id);
	// write data
	for(int i = 0; i < FF_PROC_PASS; i++){
		const uint tmp_id = i * FF_THREADS_PER_CENTER + threadIdx.x;
		const uint irow = tmp_id / PATCH_WIDTH;
		const uint icol = tmp_id % PATCH_WIDTH;
		const int drow = crow + irow - rank;
		const int dcol = ccol + icol - rank;
		if (irow < PATCH_WIDTH && status[irow][icol] == center_id && drow >= 0 && drow < HEIGHT && dcol >= 0 && dcol < WIDTH)
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
	checkCudaError(cudaMemset(d_conmap, 0, sizeof(uint)*LSIZE), "cudaMemset d_conmap");
	checkCudaError(cudaMemcpy(d_data, h_data, LSIZE * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy h2d");
}

void releaseData()
{
	cudaFree(d_data);
	cudaFree(d_conmap);
}


extern "C" void processImages(float *data, Peak *&peak_out, int &npeaks, uint *data_out)
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
	npeaks = num_pix;

	cudaEventRecord(t0);
	checkCudaError(cudaMalloc((void **)&d_peaks, num_pix * sizeof(Peak)), "cudaMalloc d_peaks");
	checkCudaError(cudaMemset(d_peaks, 0, num_pix * sizeof(Peak)), "cudaMemset d_peaks");
	floodFill_v2<<<NUM_BLOCKS, FF_THREADS_PER_BLOCK>>>(d_data, d_dense_centers, d_peaks, d_conmap);
	cudaEventRecord(t1);
	cudaEventSynchronize(t1);
	cudaEventElapsedTime(&miliseconds, t0, t1);
	printf("floodFill_v2 takes %f miliseconds\n", miliseconds);
	getCudaError("floodFill_v2");

	peak_out = new Peak[num_pix];
	cudaEventRecord(t0);
	checkCudaError(cudaMemcpy(peak_out, d_peaks, num_pix * sizeof(Peak), cudaMemcpyDeviceToHost), "cudaMemcpy d2h");
	cudaEventRecord(t1);
	cudaEventSynchronize(t1);
	cudaEventElapsedTime(&miliseconds, t0, t1);
	printf("copying peaks to cpu takes %f miliseconds\n", miliseconds);
	if (data_out != NULL)
	{
		checkCudaError(cudaMemcpy(data_out, d_conmap, LSIZE * sizeof(uint), cudaMemcpyDeviceToHost), "cudaMemcpy d2h");
	}

	releaseData();
}