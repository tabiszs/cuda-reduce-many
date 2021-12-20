#include <iostream>
#include <cstdlib>
#include "reduce_many.h"
#include "reduce_local.h"

void getNM(int *n, int *m, int argc, char** argv) {
    if(argc != 3) {
        std::cout << "Syntax: ./prog n m\n";
        exit(1);
    }
    *n = atoi(argv[1]);
    *m = atoi(argv[2]);
}

void printHeader(int n, int m) {
    std::cout << "----------Reduction Many 1.9 -----------" << std::endl;
    std::cout << "Number of array: " << n << std::endl;
    std::cout << "Number of elements in array: " << m << std::endl;
}

void printResults(int n, int m, double *h_data) {
    int min = n < 3 ? n : 3; 
    for(int i=0; i<n; ++i) {            
        if (h_data[i] == m)
            std::cout << "i: " << i << " Correct result: " << h_data[i] << std::endl;
        else 
            std::cout << "i: " << i << " Incorrect result: " << h_data[i] << std::endl;
    }
}

void printResultsForLocalReduction(int n, int m, double *h_data) {
    int min = n < 3 ? n : 3; 
    for(int i=0, index = 0; i<min; ++i, index += m) {            
        if (h_data[index] == m)
            std::cout << "Correct result: " << h_data[index] << std::endl;
        else 
            std::cout << "Incorrect result: " << h_data[index] << std::endl;
    }
}

int sumOneArray(int m, double* d_in, double* d_out) {
    int err;
    if((err=local_reductionR(d_in, m, d_out, 512, 0))!=0) {
        return err;
    }
    return 0;
}

int performTestLocal(int n, int m, int N, double* d_in, double* h_data) {
    for (int i=0; i<N; ++i) 
        h_data[i] = 1;

    cudaMemcpy(d_in, h_data, N*sizeof(double), cudaMemcpyHostToDevice);

    size_t shift = 0;
    for(int i=0; i<n; ++i) {
        sumOneArray(m, d_in+shift, d_in+shift);
        shift += m;
    }

    cudaMemcpy(h_data, d_in, N*sizeof(double), cudaMemcpyDeviceToHost);

    printResultsForLocalReduction(n,m,h_data);
    return 0;
}

int performTestMany(int n, int m, int N, double* d_in, double* h_data) {
    for (int i=0; i<N; ++i) 
        h_data[i] = 1;

    cudaMemcpy(d_in, h_data, N*sizeof(double), cudaMemcpyHostToDevice);

    int err;
    if((err=local_reductions_many(n, m, d_in, d_in))!=0) {
        return err;
    }

    cudaMemcpy(h_data, d_in, N*sizeof(double), cudaMemcpyDeviceToHost);

    printResults(n,m,h_data);
    return 0;
}


int main(int argc, char **argv) {

    int n, m, N;
    double *h_data, *d_in;

    getNM(&n, &m, argc, argv);
    N = n*m;    
    h_data = (double*)malloc(N*sizeof(double));
    cudaMalloc(&d_in, N*sizeof(double));

    printHeader(n,m);

    // Warm Up
    //for(int i=0; i<20; ++i)
    //    performTestManyOld(n, m, N, d_in, h_data);

    // Test Previous Redcution
    //performTestLocal(n, m, N, d_in, h_data);

    // Test New Reduction
    //performTestManyOld(n, m, N, d_in, h_data);

    // Thrust Reduction
    for(int i=0; i<100; i++) {
        performTestMany(n, m, N, d_in, h_data);
    }

    cudaFree(d_in);
    free(h_data);
    return 0;
}