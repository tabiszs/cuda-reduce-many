// Author: Stanisław Tabisz
// Date: 16-11-2021
// Version: 1.5

#include <iostream>

# define NTHREADS_MANY_OLD 512

__device__ unsigned int getEndValue_old(unsigned int currArrayIdx, 
    unsigned int nElementsInVector, unsigned int nMemoryBlocksPerArray, unsigned int currMemoryBlockIdx) {
    unsigned int idxLastMemoryBlockInCurrArray = (currArrayIdx+1)*nMemoryBlocksPerArray - 1;
    unsigned int nthreadsUsedInLastMemoryBlock = nElementsInVector - (nMemoryBlocksPerArray-1)*blockDim.x;
    return currMemoryBlockIdx < idxLastMemoryBlockInCurrArray ? blockDim.x : nthreadsUsedInLastMemoryBlock;
}

__device__ void warpReduce_old(volatile double *shm, int tid, int nEndValue) {
    if (tid + 32 < nEndValue) shm[tid] += shm[tid + 32];
    if (tid + 16 < nEndValue) shm[tid] += shm[tid + 16];
    if (tid + 8  < nEndValue) shm[tid] += shm[tid + 8];
    if (tid + 4  < nEndValue) shm[tid] += shm[tid + 4];
    if (tid + 2  < nEndValue) shm[tid] += shm[tid + 2];
    if (tid + 1  < nEndValue) shm[tid] += shm[tid + 1]; 
}

template <unsigned int blockSize>
__global__ void reduce_many_old(double *g_idata, double *g_odata, 
    unsigned int nElementsInVector, unsigned int nblocksPerArray, unsigned int nMemoryBlocksPerArray) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int currArrayIdx = blockIdx.x / nblocksPerArray;
    unsigned int currBlockIdx = blockIdx.x - currArrayIdx*nblocksPerArray;
    unsigned int currMemoryBlockIdx = currArrayIdx*nMemoryBlocksPerArray + 2*currBlockIdx;
    unsigned int uniqueKernelID = currMemoryBlockIdx*blockDim.x + threadIdx.x;
    unsigned int shift = nMemoryBlocksPerArray*blockDim.x - nElementsInVector;
	unsigned int idxInGlobalMemory = uniqueKernelID - (shift*currArrayIdx);
    unsigned int nEndValue = getEndValue_old(currArrayIdx, nElementsInVector, nMemoryBlocksPerArray, currMemoryBlockIdx);


    // each thread loads two element from global to shared mem
    if(2*currBlockIdx*blockDim.x + tid + blockDim.x < nElementsInVector)
        sdata[tid] = g_idata[idxInGlobalMemory] + g_idata[idxInGlobalMemory + blockDim.x];
    else
        sdata[tid] = g_idata[idxInGlobalMemory];
    __syncthreads();

    // reduce
    if (blockSize >= 512) { if (tid < 256 && tid + 256 < nEndValue) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128 && tid + 128 < nEndValue) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64  && tid + 64  < nEndValue) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }  
    if (tid < 32)
        warpReduce_old(sdata, tid, nEndValue);

    // write result for this block to global mem
    if (tid == 0) 
        g_odata[currArrayIdx*nblocksPerArray + currBlockIdx] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce_one(double *g_idata, double *g_odata, 
    unsigned int nElementsInVector, unsigned int nblocksPerArray) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int uniqueKernelID = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int currArrayIdx = blockIdx.x / nblocksPerArray;
    unsigned int totalThreadsPerArray = nblocksPerArray*blockDim.x;
    unsigned int shift = totalThreadsPerArray - nElementsInVector;
	unsigned int idxInGlobalMemory = uniqueKernelID - (shift*currArrayIdx);
    unsigned int endValue = getEndValue_old(currArrayIdx, nElementsInVector, nblocksPerArray, blockIdx.x);

    // each thread loads one element from global to shared mem
    sdata[tid] = g_idata[idxInGlobalMemory];
    __syncthreads();

    // reduce
    if (blockSize >= 512) { if (tid < 256 && tid + 256 < endValue) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128 && tid + 128 < endValue) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64  && tid + 64  < endValue) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }  
    if (tid < 32)
        warpReduce_old(sdata, tid, endValue);

    // write result for this block to global mem at idx of proceed array
    if (tid == 0) 
        g_odata[currArrayIdx] = sdata[0];
}

int getBlockNum_Many_Old(int m, int nthreads) {
    return (m - 1 + nthreads) / nthreads;
}

/**
 * Function reduces many arrays.
 * Mathematical operation: sum (+)
 * @param n number of vectors to be reduced -> vectorNo
 * @param m length of the vector            -> indexNo
 * @param d_in = A pointer to memory (device) of size n*m*sizeof(double), (INPUT)
 *          i-th element of k-th vector is located A_i^k=A[k*m + I],
 *          NOTE: the array can be overwritten by the computation process, 
 *                i.e. it can be used as working space
 * @param d_out = r pointer to memory (device) of size n*sizeof(double) (OUTPUT)
 *          where reductions will be stored, namely 
 *           r[k] = sum_i^m  A_i^{k}
 *          NOTE: it is allowed to provide as pointer r address of input array A.  
 * @return 0 – ok, otherwise error code.
 **/
int local_reductions_many_old(int n, int m, double *d_in, double *d_out) 
{
    cudaError_t error;
    const int nthreads=NTHREADS_MANY_OLD;
     error = cudaPeekAtLastError();
    if(error != cudaSuccess) 
        return -1;

    if(m <= nthreads)
    {
        unsigned int nblocksPerArray = getBlockNum_Many_Old(m, nthreads);
        unsigned int nblocksTotal = nblocksPerArray * n;
        size_t shm_size = nthreads * sizeof(double);
        reduce_one<nthreads><<<nblocksTotal, nthreads, shm_size>>>(d_in, d_out, m, nblocksPerArray);
        error = cudaGetLastError();
        if(error != cudaSuccess) 
            return error;
    }
    else
    {
        unsigned int nblocksPerArray = (m - 1 + nthreads) / nthreads;
        unsigned int optimalized_nblocksPerArray = (nblocksPerArray -1 + 2) / 2;
        unsigned int nblocksTotal = optimalized_nblocksPerArray * n;
        size_t shm_size = nthreads * sizeof(double);

        // Do first reduction
        reduce_many_old<nthreads><<<nblocksTotal, nthreads, shm_size>>>(
            d_in, d_out, m, optimalized_nblocksPerArray, nblocksPerArray);
        error = cudaGetLastError();
        if(error != cudaSuccess) 
            return error;

        // Do iteratively reduction of partial_sums
        while(optimalized_nblocksPerArray>1)
        {            
            m = optimalized_nblocksPerArray;
            nblocksPerArray = (optimalized_nblocksPerArray - 1 + nthreads) / nthreads;
            optimalized_nblocksPerArray = (nblocksPerArray -1 + 2) / 2;
            nblocksTotal = optimalized_nblocksPerArray*n;
            reduce_many_old<nthreads><<<nblocksTotal, nthreads, shm_size>>>(
                d_out, d_out, m, optimalized_nblocksPerArray, nblocksPerArray);
            error = cudaGetLastError();
            if(error != cudaSuccess) 
                return error;
        }        
    }
    return 0;
}