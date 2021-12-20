#ifndef REDUCE_MANY_H
#define REDUCE_MANY_H
  
template <unsigned int blockSize>
void reduce_many(double *g_idata, double *g_odata, unsigned int nElementsInVector, 
    unsigned int nblocksPerArray, unsigned int nMemoryBlocksPerArray);
    
template <unsigned int blockSize>
void reduce_one(double *g_idata, double *g_odata, unsigned int nElementsInVector, unsigned int nblocksPerArray);

unsigned int getNoEndValue(unsigned int idxcurrArrayIdx, unsigned int nElementsInVector, unsigned int nblocksPerArray);
void warpReduce(volatile int *shm, int tid, int endVal);
int local_reductions_many(int n, int m, double *d_in, double *d_out);

#endif