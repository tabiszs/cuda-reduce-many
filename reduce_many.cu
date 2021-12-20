// Author: Stanisław Tabisz
// Date: 25-11-2021
// Version: 1.7

#include <iostream>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>

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
int local_reductions_many(int n, int m, double *d_in, double *d_out) {

    cudaError_t error = cudaPeekAtLastError();
    if(error != cudaSuccess) 
        return -1;

    thrust::device_vector<short> keys(n*m);
    for(int i=1; i<n; i+=2) {
        thrust::fill(thrust::device, keys.begin() + i*m, keys.begin() + i*m + m, 1);
    }

    //short *raw_keys_ptr = thrust::raw_pointer_cast(&keys[0]);
    thrust::device_ptr<double> d_in_ptr(d_in);
    thrust::device_ptr<double> d_out_ptr(d_out);
    thrust::reduce_by_key(thrust::device, keys.begin(), keys.end(), d_in_ptr, keys.begin(), d_out_ptr);
    return 0;
}