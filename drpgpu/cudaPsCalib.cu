#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <unistd.h>

#include <sys/time.h>
#include <iostream>
#include <iomanip>
using namespace std;

#include <string>
#include <sstream>
#include <fstream>

#include "cudaPsCalib.h"

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__device__ void reduce(float *sdata)
{
  int tid = threadIdx.x;
  
  // do reduction in shared mem
  for(int s=1; s < blockDim.x; s*=2){
    
    int index = 2 * s * tid;

    if (index < blockDim.x) {
      // if a block is not a multiple of two, leave as-is
      if (index + s < blockDim.x)
        sdata[index] += sdata[index + s];
    }

    __syncthreads();
  }

}

/* -------------------------- calibration kernels ------------------------------*/

__global__ void kernel(float *a, 
                        int offset,
                        float *dark,
                        short *bad,
                        float cmmThr,
                        int streamSize,
                        float *blockSum, 
                        int *cnBlockSum){
  __shared__ float sdata[N_COLS];
  __shared__ float scount[N_COLS];

  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int iData = offset + idx;
  int iDark = iData % N_PIXELS;
  
  // subtrack pedestal and only use data with flag good pixel (bad==1)
  // and with value below a user-specified threshold.
  a[iData] -= dark[iDark];
  a[iData] *= bad[iDark];
  sdata[tid] = a[iData] * (a[iData] < cmmThr);
  scount[tid] = 1.0f * bad[iDark] * (a[iData] < cmmThr);

  __syncthreads();
 
  // calculate blocksum and blockcount
  reduce(sdata);
  reduce(scount);
  
  // save results - calculate block id using offset
  if (tid == 0){
    int iBlock = floor( (double) iData / blockDim.x );
    blockSum[iBlock] = sdata[0];
    cnBlockSum[iBlock] = (int)scount[0];
  }
  
}


__global__ void common_mode(float *blockSum, int *cnBlockSum, float *sectorSum, int *cnSectorSum, int offset)
{
  // calculate sector sum and sector count
  __shared__ float s_blockSum[N_ROWS];
  __shared__ float s_cnBlockSum[N_ROWS];

  int tid = threadIdx.x;
  int iBlock = tid + offset;
  s_blockSum[tid] = blockSum[iBlock];
  s_cnBlockSum[tid] = (float)cnBlockSum[iBlock];

  __syncthreads();

  reduce(s_blockSum);
  reduce(s_cnBlockSum);
  
  // save results - calculate sector id using offset
  if (tid == 0){
    int iSector = floor( (double) iBlock / blockDim.x );
    sectorSum[iSector] = s_blockSum[0];
    cnSectorSum[iSector] = (int)s_cnBlockSum[0];
  }
  
}

__global__ void common_mode_apply(float *a, float *sectorSum, int *cnSectorSum, float *gain, int offset)
{
  int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
  int iGain = i % N_PIXELS;
  int iSector = floor( (double) i / SECTOR_SIZE );
  a[i] = ( a[i] - (sectorSum[iSector] / cnSectorSum[iSector]) ) * gain[iGain];
}
   

/* ---------------------- host code -----------------------------*/
void write_file(string fileName, float *data, int n)
{
  FILE *pFile = fopen(fileName.c_str(), "w");
  if (pFile)
  {
    for (int i=0; i<n; i++)
      fprintf(pFile, "%f\n", data[i]);
  }
  fclose(pFile);
}

void fill( float *p, int n, float val ) {
  for(int i = 0; i < n; i++){
    p[i] = val;
  }
}

float maxError(float *aCalc, float *aKnown, int nEvents)
{
  float maxE = 0;
  for (int i = 0; i < nEvents; i++) {
    int offset = i * N_PIXELS;
    for (int j = 0; j < N_PIXELS; j++) {
      int idx = offset + j;
      float error = fabs(aCalc[idx]-aKnown[j]);
      //if (error > 5.0)
      //  printf("offset: %d j: %d idx: %d error %e aCalc[idx]: %8.2f aKnown[j]: %8.2f\n", offset, j, idx, error, aCalc[idx], aKnown[j]);
      if (error > maxE) maxE = error;
    }
  }
  return maxE;
}

// used in host_calculation qsort function
int compare (const void * a, const void * b)
{
  float fa = *(const float*) a;
  float fb = *(const float*) b;
  return (fa > fb) - (fa < fb);
}

// host-side calculation comparision
void host_calc(float *a, float *dark, float cmmThr) {
  // host calculation
  struct timeval start, end;

  long seconds, useconds;
  double mtime;

  gettimeofday(&start, NULL);
  
  // dark
  for(int i = 0; i < N_PIXELS; i++)
    a[i] -= dark[i];

  // common mode 
  float *sectorMedian = (float *)malloc(MAX_QUADS * MAX_SECTORS * sizeof(float));
  for (int i = 0; i < MAX_QUADS * MAX_SECTORS; i++) {
    
    int offset = i * SECTOR_SIZE;
    
    // select only this sector and sort this sector
    float *sector = (float *)malloc(SECTOR_SIZE * sizeof(float));
    for (int j = 0; j < SECTOR_SIZE; j++) {
      sector[j] = a[offset + j]; 
    }

    //printf("\n");
    //printf("s[0]=%6.2f, s[1]=%6.2f, s[2]=%6.2f\n", sector[0], sector[1], sector[2]);
    
    qsort(sector, SECTOR_SIZE, sizeof(float), compare);
    //printf("%6.2f, %6.2f, %6.2f ... %6.2f, %6.2f, %6.2f\n", sector[0], sector[1], sector[2], sector[SECTOR_SIZE-3], sector[SECTOR_SIZE-2], sector[SECTOR_SIZE-1]);
    
    // apply the threshold
    int foundPos = 0;
    for (int j = SECTOR_SIZE - 1; j >= 0; j--) {
      if (sector[j] <= cmmThr) {
        foundPos = j;
        break;
      }
      if (j == 0) foundPos = SECTOR_SIZE - 1;
    }   
    
    // calculate median
    if(foundPos%2 == 0) {
      sectorMedian[i] = (sector[foundPos/2] + sector[foundPos/2 - 1]) / 2.0;
    } else {
      sectorMedian[i] = sector[foundPos/2];
    } 
    free(sector);
    printf("sector: %d foundPos: %d med: %6.4f \n", i, foundPos, sectorMedian[i]); 
    
  }

  // apply common mode
  for(int i=0; i < N_PIXELS; i++) {
    int iSector = floor(i / SECTOR_SIZE);
    a[i] -= sectorMedian[iSector];
  }
  
  gettimeofday(&end, NULL);

  seconds  = end.tv_sec  - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;
  mtime = ((seconds) * 1000000 + useconds)/1000.0;// + 0.5;

  cout << "Host dark-subtraction and common mode took "<< mtime <<" ms for 1 event."<< endl;
}

int main(int argc, char **argv)
{
  const int nEvents = atoi(argv[1]);			        // no. of events
  const int n = N_PIXELS * nEvents;			          // total number of pixels
  
  const int blockSize = N_COLS;                   // block size is set to no. of columns in a sector

  const int bytes = n * sizeof(float);			      // total size (bytes)
  const int darkBytes = N_PIXELS * sizeof(float);	// dark size (bytes)

  const int nBlocks = N_ROWS * nEvents; 
  const int blockSumBytes = nBlocks * sizeof(float);

  const int nSectors = MAX_QUADS * MAX_SECTORS * nEvents;
  const int sectorSumBytes = nSectors * sizeof(float);

  const float cmmThr = 10.0f;
  
  int devId = 0;
  if (argc > 2) devId = atoi(argv[2]);			     // device ID (optional)
  
  // print device name
  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda( cudaSetDevice(devId) );

  // allocate pinned host memory and device memory
  // RAW * nEVents
  float *a, *d_a; 						
  checkCuda( cudaMallocHost((void**)&a, bytes) );    // host pinned
  checkCuda( cudaMalloc((void**)&d_a, bytes) );  // device data only allocate enough for 1 evt
  // SINGLE RAW
  float *raw;                                               
  checkCuda( cudaMallocHost((void**)&raw, darkBytes) );     
  // PEDESTAL
  float *dark, *d_dark;					 
  checkCuda( cudaMallocHost((void**)&dark, darkBytes) ); 	
  checkCuda( cudaMalloc((void**)&d_dark, darkBytes) );		
  // PER-PIXEL GAIN
  float *gain, *d_gain;					 	
  checkCuda( cudaMallocHost((void**)&gain, darkBytes) ); 	
  checkCuda( cudaMalloc((void**)&d_gain, darkBytes) );		
  // BAD PIXEL FLAGS
  short *bad, *d_bad;           
  checkCuda( cudaMallocHost((void**)&bad, N_PIXELS * sizeof(short)) );  
  checkCuda( cudaMalloc((void**)&d_bad, N_PIXELS * sizeof(short)) );    
  // CALIBRATED
  float *calib, *d_calib;					 	
  checkCuda( cudaMallocHost((void**)&calib, darkBytes) ); 	
  checkCuda( cudaMalloc((void**)&d_calib, darkBytes) );		  
  // Sum of each block
  float *d_blockSum, *blockSum; 
  checkCuda( cudaMalloc((void**)&d_blockSum, blockSumBytes) );
  checkCuda( cudaMallocHost((void**)&blockSum, blockSumBytes) );
  cudaMemset(d_blockSum, 0, blockSumBytes);
  int *d_cnBlockSum, *cnBlockSum;
  checkCuda( cudaMalloc((void**)&d_cnBlockSum, nBlocks * sizeof(int)) );
  checkCuda( cudaMallocHost((void**)&cnBlockSum, nBlocks * sizeof(int)) );
  cudaMemset(d_cnBlockSum, 0, nBlocks * sizeof(int));
  // Sum of each sector
  float *d_sectorSum, *sectorSum; 
  checkCuda( cudaMalloc((void**)&d_sectorSum, sectorSumBytes) );
  checkCuda( cudaMallocHost((void**)&sectorSum, sectorSumBytes) );
  cudaMemset(d_sectorSum, 0, sectorSumBytes);
  int * d_cnSectorSum;
  checkCuda( cudaMalloc((void**)&d_cnSectorSum, nSectors * sizeof(int)) );
  // Peak centroids - allocate for all events
  // 8 centers per patch, 
  // 12x47=564  patches per sector, 
  // 564x8=4512 centers per sector
  // 4512x32 = 144384 centers per event.
  const int nCentersPerSector = FILTER_PATCH_PER_SECTOR * (FILTER_PATCH_WIDTH / FILTER_PATCH_HEIGHT);
  const int nCentersPerEvent = nCentersPerSector * MAX_QUADS * MAX_SECTORS;
  const int nCenters = nCentersPerEvent * nEvents;
  uint *d_centers, *centers;
  checkCuda( cudaMalloc((void**)&d_centers, nCenters * sizeof(uint)) );
  checkCuda( cudaMallocHost((void**)&centers, nCenters * sizeof(uint)) );
  checkCuda( cudaMemset(centers, 0, nCenters * sizeof(uint)) );
  checkCuda( cudaMemset(d_centers, 0, nCenters * sizeof(uint)));
  // Peaks - peak is allocated for all events since we need to copy
  // peaks for each event out.
  int nPeaks = nCenters;  
  Peak *d_peaks, *peaks;
  checkCuda( (cudaMalloc((void**)&d_peaks, nPeaks * sizeof(Peak))) );
  checkCuda( (cudaMallocHost((void**)&peaks, nPeaks * sizeof(Peak))) );
  checkCuda( (cudaMemset(d_peaks, 0, nPeaks * sizeof(Peak))) );
  checkCuda( (cudaMemset(peaks, 0, nPeaks * sizeof(Peak))) );
  uint *d_conmap;
  checkCuda( (cudaMalloc((void**)&d_conmap, n * sizeof(uint))) );
  checkCuda( (cudaMemset(d_conmap, 0, n * sizeof(uint))) );

  //load the text file and put it into a single string:
  ifstream inR("data/cxid9114_r95_evt01_raw.txt");
  ifstream inP("data/cxid9114_r95_evt01_ped.txt");
  ifstream inG("data/cxid9114_r95_evt01_gmap.txt");
  ifstream inB("data/cxid9114_r95_evt01_stmask.txt"); // 0 - bad, 1 - Good
  ifstream inC("data/cxid9114_r95_evt01_calib.txt");
  
  // Fill arrays from text files
  string line;
  for (unsigned int i=0; i<N_PIXELS; i++){
    getline(inR, line);
    raw[i] = atof(line.c_str());
    getline(inP, line);
    dark[i] = atof(line.c_str());
    getline(inG, line);
    gain[i] = atof(line.c_str());
    getline(inB, line);
    bad[i] = atoi(line.c_str());
    getline(inC, line);
    calib[i] = atof(line.c_str());
    //populate all events with the same set of test data
    for (int j=0; j<nEvents; j++) {
      int offset = j * N_PIXELS;
      a[offset + i] = raw[i];
    }
  }
  puts("Input\n");
  printf("Data       : %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", a[0], a[1], a[2], a[n-3], a[n-2], a[n-1]);
  printf("Dark       : %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", dark[0], dark[1], dark[2], dark[N_PIXELS-3], dark[N_PIXELS-2], dark[N_PIXELS-1]);
  printf("Bad pixels : %d %d %d...%d %d %d\n", bad[0], bad[1], bad[2], bad[N_PIXELS-3], bad[N_PIXELS-2], bad[N_PIXELS-1]);
  printf("Pixel gain : %8.2f %8.2f %8.2f ... %8.2f %8.2f %8.2f\n", gain[0], gain[1], gain[2], gain[N_PIXELS-3], gain[N_PIXELS-2], gain[N_PIXELS-1]);
  

  // host calculation 
  /*host_calc(raw, dark, cmmThr);

  
  printf("Host Calculation\n");
  printf("Input values (Data calc.): %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", raw[0], raw[1], raw[2], raw[N_PIXELS-3], raw[N_PIXELS-2], raw[N_PIXELS-1]);
  printf("Input values (Data known): %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", calib[0], calib[1], calib[2], calib[N_PIXELS-3], calib[N_PIXELS-2], calib[N_PIXELS-1]);
  printf("  max error: %e\n", maxError(raw, calib, 1));
  */

  // 
  // serial copy for one dark, bad pixel mask, and pixel gain to device 
  checkCuda( cudaMemcpy(d_dark, dark, darkBytes, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(d_bad, bad, N_PIXELS * sizeof(short), cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(d_gain, gain, darkBytes, cudaMemcpyHostToDevice) );

  float ms; // elapsed time in milliseconds

  // create events and streams
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[N_STREAMS];
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  checkCuda( cudaEventCreate(&dummyEvent) );
  for (int i = 0; i < N_STREAMS; ++i)
    checkCuda( cudaStreamCreate(&stream[i]) );

  // asynchronous version 1: loop over {copy, kernel, copy}
  checkCuda( cudaEventRecord(startEvent, 0) );
  cudaProfilerStart();
  for (int evt = 0; evt < nEvents; evt++) {
    // Each event is divided into 32 streams
    int evtOffset = evt * N_PIXELS;
    for (int s=0; s < N_STREAMS; s++){
      // For copying data in, the offset is calculated from evt#
      int streamSize = ceil( (double) N_PIXELS / N_STREAMS );
      int offset = evtOffset + (s * streamSize);
      int streamBytes = streamSize * sizeof(float);   
      int gridSize = ceil(  (double) streamSize / blockSize );             

      checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset],
                                 streamBytes, cudaMemcpyHostToDevice,
                                 stream[s]) );

      // calibration kernels
      kernel<<<gridSize, blockSize, 0, stream[s]>>>(d_a, offset, d_dark, d_bad, cmmThr, streamSize, d_blockSum, d_cnBlockSum);
    
      // Common mode kernel reduce blockSum to sectorSum
      // We use 388 threads to reduce 388 blockSum (or sum of each row)
      // to a sector sum. No. of blocks is then equal to the no. of events.
      int cmmOffset = (evt * N_ROWS * N_STREAMS) + (s * N_ROWS);
      common_mode<<<1, N_ROWS, 0, stream[s]>>>(d_blockSum, d_cnBlockSum, d_sectorSum, d_cnSectorSum, cmmOffset);
      common_mode_apply<<<gridSize, blockSize, 0, stream[s]>>>(d_a, d_sectorSum, d_cnSectorSum, d_gain, offset); 

      // peakFinder kernels
      int filterOffset = (evt * FILTER_PATCH_PER_SECTOR * N_STREAMS) + (s * FILTER_PATCH_PER_SECTOR);
      filterByThrHigh_v2<<<FILTER_PATCH_PER_SECTOR, FILTER_THREADS_PER_PATCH, 0, stream[s]>>>(d_a, d_centers, filterOffset);
      
      // floodFill kernel is activiated by sending 64 threads to work
      // on each center.
      int ffOffset = (evt * nCentersPerSector * N_STREAMS) + (s * nCentersPerSector);
      floodFill_v2<<<nCentersPerSector, FF_LOAD_THREADS_PER_CENTER, 0, stream[s]>>>(d_a, d_centers, d_peaks, d_conmap, ffOffset);

      // copy data out
      checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset],
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[s]) );

      // copy peaks out
      checkCuda( cudaMemcpyAsync(&peaks[ffOffset], &d_peaks[ffOffset],
                               nCentersPerSector * sizeof(Peak), cudaMemcpyDeviceToHost,
                               stream[s]) ); 
    }
  }
  cudaProfilerStop(); 
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("GPU Calculation\n");
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("GPU Calibrated   : %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", a[0], a[1], a[2], a[n-3], a[n-2], a[n-1]);
  printf("CPU Calibrated   : %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", calib[0], calib[1], calib[2],calib[N_PIXELS-3], calib[N_PIXELS-2], calib[N_PIXELS-1]);
  printf("Differences      : %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", a[0]-calib[0], a[1]-calib[1], a[2]-calib[2], a[n-3]-calib[N_PIXELS-3], a[n-2]-calib[N_PIXELS-2], a[n-1]-calib[N_PIXELS-1]);
  printf("  max error      : %e\n", maxError(a, calib, nEvents));
     
  //write_file("calc_calib.txt", a, n);

  /*cudaMemcpy(sectorSum, d_sectorSum, nSectors * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < nSectors; i++) {
    printf("i=%d sectorSum[i]=%f\n", i, sectorSum[i]);
  }*/
  
  /*int cnNonZeroCenters = 0;
  checkCuda( cudaMemcpy(centers, d_centers, nCenters * sizeof(uint), cudaMemcpyDeviceToHost) );
  for (int i=0; i < nCentersPerEvent * nEvents; i++){
    if (centers[i] != 0){
      int sectorId1 = (float) centers[i] / SECTOR_SIZE;
      int sectorId2 = (float) i / (nCentersPerSector);
      printf("i: %d, centers[i]:%d sectorByPixel: %d val: %6.2f sectorByIndex:%d\n", i, centers[i], a[centers[i]], sectorId1, sectorId2);
      cnNonZeroCenters++;
    }
  }
  printf("Total non zero centers: %d\n", cnNonZeroCenters);*/

  printf("nCenters: %d, nPeaks: %d, FFP_SECTOR: %d, W: %d, H: %d, nSectors: %d\n", nCenters,
            nPeaks, FILTER_PATCH_PER_SECTOR, FILTER_PATCH_WIDTH, FILTER_PATCH_HEIGHT, nSectors);

  int cnValidPeaks = 0;
  for (int i=0; i < nPeaks; i++) {
    if (peaks[i].valid) {
      //printf("i: %d, evt: %d, sector: %d, row: %d, col: %d\n", i, (int)peaks[i].evt, (int)peaks[i].seg, (int)peaks[i].row, (int)peaks[i].col);
      cnValidPeaks++;
    }
  }
  printf("nValidPeaks: %d\n", cnValidPeaks);
  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  checkCuda( cudaEventDestroy(dummyEvent) );
  for (int i = 0; i < N_STREAMS; ++i)
    checkCuda( cudaStreamDestroy(stream[i]) );
  cudaFree(d_a);
  cudaFreeHost(a);
  cudaFreeHost(raw);
  cudaFree(d_dark);
  cudaFreeHost(dark);
  cudaFree(d_gain);
  cudaFreeHost(gain);
  cudaFree(d_calib);
  cudaFreeHost(calib);
  cudaFree(d_blockSum);
  cudaFreeHost(blockSum);
  cudaFree(d_cnBlockSum);
  cudaFreeHost(cnBlockSum);
  cudaFree(d_sectorSum);
  cudaFreeHost(sectorSum);
  cudaFree(d_cnSectorSum);
  cudaFree(d_centers);
  cudaFreeHost(centers);
  cudaFree(d_peaks);
  cudaFreeHost(peaks);
  cudaFree(d_conmap);
  
  cudaDeviceReset();

  return 0;
}
