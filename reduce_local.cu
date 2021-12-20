// Author: Gabriel Wlazlowski
// Date: 09-09-2016
// From: pca_kernels.cu

////////////////////////////////////////////////////////////////////////////////
// LOCAL REDUCTIONS
////////////////////////////////////////////////////////////////////////////////
int opt_threads(int new_blocks,int threads, int current_size)
{
    int new_threads;
    if(new_blocks==1) 
    {
        new_threads=2; 
        while(new_threads<threads) 
        { 
            if(new_threads>=current_size) break;
            new_threads*=2;
        }
    }
    else new_threads=threads;
    return new_threads;
}

template <unsigned int blockSize>
__device__ void warpReduceR(volatile double *sdata, unsigned int tid) 
{
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
    if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
    if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
    if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
}

template <unsigned int blockSize>
__global__ void __reduce_kernelR__(double *g_idata, double *g_odata, int num_wf, int mode)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int ishift=i+blockDim.x;
    
    // Loading data
//     if(mode==0) // sum of doubles
    {
        if(ishift<num_wf) sdata[tid] = g_idata[i] + g_idata[ishift];
        else if(i<num_wf) sdata[tid] = g_idata[i];
        else         sdata[tid] = 0.0;
    }
//     else // add here other modes
    

    __syncthreads();
    
    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >=  512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >=  256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >=  128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
    if (tid < 32) warpReduceR<blockSize>(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void call_reduction_kernelR(int dimGrid, int dimBlock, int size, double *d_idata, double *d_odata, int mode)
{
    int smemSize=dimBlock*sizeof(double);
    switch (dimBlock)
    {
        case 1024:
            __reduce_kernelR__<1024><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, mode); break;
        case 512:
            __reduce_kernelR__< 512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, mode); break;
        case 256:
            __reduce_kernelR__< 256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, mode); break;
        case 128:
            __reduce_kernelR__< 128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, mode); break;
        case 64:
            __reduce_kernelR__<  64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, mode); break;
    }   
}


int getBlockNum(int size, int threads) {
    return (int)ceil((float)size/threads);
}

/**
 * Function does fast reduction (sum of elements) of array. 
 * Result is located in partial_sums[0] element
 * If partial_sums==array then array will be destroyed
 * @param mode 0: add numbers (no transformation)
 * */ 
extern "C" int local_reductionR(double *array, int size, double *partial_sums, int threads, int mode)
{
    int blocks=getBlockNum(size, threads);
    unsigned int lthreads=threads/2; // Threads is always power of 2
    if(lthreads<64) lthreads=64; // at least 2*warp_size
    unsigned int new_blocks, current_size;

    // First reduction of the array
    call_reduction_kernelR(blocks, lthreads, size, array, partial_sums, mode);
    
    // Do iteratively reduction of partial_sums
    current_size=blocks;
    while(current_size>1)
    {
        new_blocks=getBlockNum(current_size, threads);
        lthreads=opt_threads(new_blocks,threads, current_size)/2;
        if(lthreads<64) lthreads=64; // at least 2*warp_size
        call_reduction_kernelR(new_blocks, lthreads, current_size, partial_sums, partial_sums, 0);
        current_size=new_blocks;
    }    
    
    return 0;
}
