#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <unordered_set>
#include "peakFinder.h"

// static std::string FILE_NAME = "../regent/small_test";
static std::string FILE_NAME = "/reg/d/psdm/cxi/cxitut13/scratch/cpo/test1000.bin";
bool WRITE_FILE = true;
bool WRITE_PEAKS = false;

void load_file(std::string file_name, float *data)
{
	FILE *pFile = fopen(file_name.c_str(), "r");
	if (pFile)
	{
		long result = fread(data, sizeof(float), LSIZE, pFile);
		if (result != LSIZE)
		{
			std::cout << "result != lsize" << std::endl;
		}
	}
	else
	{
		std::cout << "cannot find " << file_name << std::endl;
	}
	fclose(pFile);
}

void write_file(std::string file_name, unsigned int *data)
{
	float *data_write = new float[LSIZE];
	// std::unordered_set<uint> non_zeros;
	for (int i = 0; i < LSIZE; i++)
	{
		if (data[i] > 0) {
			data_write[i] = 1;
			// non_zeros.insert(data[i]);
		}
		else data_write[i] = 0;
	}
	FILE *pFile = fopen(file_name.c_str(), "w");
	if (pFile)
	{
		fwrite(data_write, sizeof(float), LSIZE, pFile);
	}
	fclose(pFile);
}

void write_peaks(std::string file_name, Peak *peaks)
{
	FILE *pFile = fopen(file_name.c_str(), "w");
	if (pFile)
	{
		char hdr[] = "Evt Seg  Row  Col  Npix      Amax      Atot   rcent   ccent rsigma  csigma rmin rmax cmin cmax    bkgd     rms     son\n";
		for (int i = 0; i < MAX_PEAKS; i++)
		{
			Peak peak = peaks[i];
			if (peak.valid)
			{
				fprintf(pFile, "%3d %3d %4d %4d  %4d  %8.1f  %8.1f  %6.1f  %6.1f %6.2f  %6.2f %4d %4d %4d %4d  %6.2f  %6.2f  %6.2f\n", int(peak.evt),
          (int)(peak.seg), (int)(peak.row), (int)(peak.col), (int)(peak.npix), peak.amp_max, peak.amp_tot, peak.row_cgrav, peak.col_cgrav, peak.row_sigma, peak.col_sigma, 
          (int)(peak.row_min), (int)(peak.row_max), (int)(peak.col_min), (int)(peak.col_max), peak.bkgd, peak.noise, peak.son);
			}
		}
	}
}

int main()
{
	float *data = new float[LSIZE];
	load_file(FILE_NAME, data);
	unsigned int *data_out = NULL;
	Peak *peak_out = NULL;
	if (WRITE_FILE)
	{
		data_out = new unsigned int[LSIZE];
	}
	if (WRITE_PEAKS)
	{
		peak_out = new Peak[MAX_PEAKS];
	}
	processImages(data, peak_out, data_out);
	if (WRITE_FILE)
	{
		write_file("peaks.img", data_out);
		delete[] data_out;
	}
	if (WRITE_PEAKS)
	{
		write_peaks("peaks.txt", peak_out);
		delete[] peak_out;
	}

	delete[] data;

}