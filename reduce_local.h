#ifndef REDUCE_LOCAL_H
#define REDUCE_LOCAL_H

int opt_threads(int new_blocks,int threads, int current_size);

template <unsigned int blockSize>
void warpReduceR(volatile double *sdata, unsigned int tid);

template <unsigned int blockSize>
void __reduce_kernelR__(double *g_idata, double *g_odata, int num_wf, int mode);

void call_reduction_kernelR(int dimGrid, int dimBlock, int size, double *d_idata, double *d_odata, int mode);
int getBlockNum(int size, int threads);
extern "C" int local_reductionR(double *array, int size, double *partial_sums, int threads, int mode);

#endif