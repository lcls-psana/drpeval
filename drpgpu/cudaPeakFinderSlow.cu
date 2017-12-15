#include <stdio.h>
#include <stdlib.h>
#include "cudaPsCalib.h"

__global__ void filterByThrHighSlow(const float *d_data, uint *d_conmap, uint sectorId)
{
  uint pixId = (blockIdx.x % BLOCKS_PER_SHOT) * blockDim.x + threadIdx.x;
  
  // check if the pixel is above the threshold and records 
  // its position in d_conmap
  if (pixId < WIDTH * HEIGHT) 
  {
    uint dataId = sectorId * WIDTH * HEIGHT + pixId;
    if (d_data[dataId] > thr_high)
      d_conmap[dataId] = dataId;
  }
}

__global__ void floodFill(const float *d_data, uint* d_conmap, bool *d_stop, bool *d_acc_stop, uint sectorId)
{
  if (d_stop[sectorId]) return;
  
  uint pixId = (blockIdx.x % BLOCKS_PER_SHOT) * blockDim.x + threadIdx.x;
  
  if (pixId < WIDTH * HEIGHT)
  {
    uint 
