
#ifndef PEAK_FINDER_H
#define PEAK_FINDER_H
#include <string>

const long EVENTS = 100;
// const long EVENTS = 5;
const int MAX_PEAKS = 150;
const long SHOTS = 32;
const long WIDTH = 388;
const long HEIGHT = 185;
const long LSIZE = EVENTS * SHOTS * WIDTH * HEIGHT;

const int rank = 4;
const float thr_high = 150;
const float thr_low = 10;
const float r0 = 5;
const float dr = 0.05;
const int HALF_WIDTH = (int)(r0 + dr);
const float peak_npix_min = 2;
const float peak_npix_max = 50;
const float peak_amax_thr = 10;
const float peak_atot_thr = 20;
const float peak_son_min = 5;

struct Peak{
  bool valid;
  float evt;
  float seg;
  float row;
  float col;
  float npix;
  float npos;
  float amp_max;
  float amp_tot;
  float row_cgrav; 
  float col_cgrav;
  float row_sigma;
  float col_sigma;
  float row_min;
  float row_max;
  float col_min;
  float col_max;
  float bkgd;
  float noise;
  float son;
};

struct Win{
	int top;
	int bot;
	int left;
	int right;
  Win():top(0),bot(HEIGHT),left(0),right(WIDTH){}
};

extern "C" void processImages(float *data, Peak *&peaks_out, int &npeaks, unsigned int *data_out);

#endif