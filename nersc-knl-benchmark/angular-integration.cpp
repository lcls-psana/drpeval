#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <mpi.h>
using namespace std;

const long N = 1024;
const long IMG_SIZE = N*N;

class AngularIntegrator
{
public:
    AngularIntegrator(int xcenter, int ycenter, vector<float>& radial_bins);
    ~AngularIntegrator();
    void operator() (float* img, float* hist);
private:
    int* m_bin_indices;
    float* m_norm;
    size_t m_norm_size;
};

AngularIntegrator::AngularIntegrator(int xcenter, int ycenter, vector<float>& radial_bins)
{
    posix_memalign((void**)&m_bin_indices, 64, sizeof(int)*IMG_SIZE);
    size_t index = 0;
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            float x = i - xcenter;
            float y = j - ycenter;
            float radius = sqrtf(x*x + y*y);
            auto it = lower_bound(radial_bins.begin(), radial_bins.end(), radius);
            m_bin_indices[index] = distance(radial_bins.begin(), it);
            index++;
        }
    }
    m_norm_size = radial_bins.size()+1;
    posix_memalign((void**)&m_norm, 64, sizeof(float)*m_norm_size);
    // bincount
    for (size_t i=0; i<IMG_SIZE; i++) {
        m_norm[m_bin_indices[i]] += 1;
    }
}

AngularIntegrator::~AngularIntegrator()
{
    free(m_bin_indices);
    free(m_norm);
}

void AngularIntegrator::operator() (float* img, float* hist)
{
    int nbins = m_norm_size - 1;
    posix_memalign((void**)&hist, 64, sizeof(float)*nbins);

    #pragma omp simd
    #pragma vector aligned
    for (size_t i=0; i<IMG_SIZE; i++) {
        hist[m_bin_indices[i]] += img[i];
    }

    #pragma omp simd
    #pragma vector aligned
    for (size_t i=0; i<nbins; i++) {
        hist[i] /= m_norm[i];
    }
}

int main(int argc, char *argv[])
{
    int rank, nprocs;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int nimages = 30;
    vector<float*> images(nimages);

    const int nbins = N - 100;
    vector<float> radial_bins(nbins);
    for (int i=0; i<nbins; i++) {
        radial_bins[i] = i + 1.0f;
    }

    AngularIntegrator integrator(N/2, N/2, radial_bins);

    float* hist;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0, 1000);

    for (int i=0; i<nimages; i++) {
        posix_memalign((void**)&images[i], 64, sizeof(float)*IMG_SIZE);
         for (size_t j=0; j<IMG_SIZE; j++) {
             images[i][j] = dis(gen);
         }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int i=0; i<nimages; i++) {
        integrator(images[i], hist);
        free(hist);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        double run_time = end - start;
        double mem_size = double(IMG_SIZE*4*nimages*nprocs) / (1024*1024*1024);
        cout<<mem_size<< " GB"<<endl;
        cout<<nprocs<<" procs"<<endl;
        cout<<run_time<<endl;
        cout<<mem_size / run_time<< " GB/s"<<endl;
        ofstream out(argv[1], ofstream::app);
        out<<nprocs<<" "<<run_time<<endl;
        out.close();
    }

    for (int i=0; i<nimages; i++) {
        free(images[i]);
    }
    MPI_Finalize();
}
