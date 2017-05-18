#include <stdlib.h>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <strings.h>

static const unsigned SIZE = 32*185*388;
static const unsigned NITER= 100;

int main() {
  std::clock_t start;
  int bytes=SIZE*sizeof(float);
  double duration;
  float* calib = (float*)malloc(bytes);
  float* bkgd = (float*)malloc(bytes);
  bzero(calib,bytes);
  bzero(bkgd,bytes);
  float thresh = 10.1;

  start = std::clock();
  unsigned nabovethresh=0;
  float* c_ptr_end = calib+SIZE;
  for (int ni=0; ni<NITER; ni++) {
    float* c_ptr = calib;
    float* b_ptr = bkgd;
    while (c_ptr<c_ptr_end) {
      float val = *c_ptr++ - *b_ptr++;
      if (val>thresh) nabovethresh++;
    }
  }

  duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC / NITER;

  std::cout<<"time: "<< duration << '\n';

  return 0;
}
