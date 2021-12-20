#ifndef REDUCE_MANY_OLD_H
#define REDUCE_MANY_OLD_H

template <unsigned int blockSize>
void reduce_many_old(double *g_idata, double *g_odata, int nElementsInVector, 
    unsigned int nblocksPerArray, unsigned int nMemoryBlocksPerArray);
    
template <unsigned int blockSize>
void reduce_one(double *g_idata, double *g_odata, int nElementsInVector, unsigned int nblocksPerArray);

unsigned int getNoEndValue_old(unsigned int idxcurrArrayIdx, unsigned int nElementsInVector, unsigned int nblocksPerArray);
void warpReduce_old(volatile double *shm, int tid, int endVal);
int getBlockNum_Many_Old(int m, int nthreads);
int local_reductions_many_old(int n, int m, double *d_in, double *d_out);

#endif