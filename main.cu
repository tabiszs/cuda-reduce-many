#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include "main.h"
#include "tests.h"
#include "reduce_many.h"
#include "reduce_local.h"
#include "reduce_many_old.h"

const std::string programVersion = "1.7";
const std::string algorithmLocal = "Reduce local algorithm";
const std::string algorithmMany1 = "Reduce many algorithm 1.5";
const std::string algorithmMany2 = "Reduce many algorithm 1.9";
const std::string dataType = "double";
const double number = 1;

int main(int argc, char **argv) {

    int ntest;
    int n =1, m = 4000000;
    int N = n*m;
    char deviceName[100];
    double *h_data, *d_in, *d_out;

    getArgs(&ntest, deviceName, argc, argv);
    h_data = (double*)malloc(N*sizeof(double));
    cudaMalloc(&d_in, N*sizeof(double));
    cudaMalloc(&d_out, N*sizeof(double));
    std::string outputFileName = getFileName(deviceName);
    std::ofstream csvFile(outputFileName, std::ofstream::out);

    Record record = {}; 
    record.deviceName = deviceName;
    record.programVersion = programVersion;
    record.dataType = dataType;
    record.numberInCell = number;
    record.inOutTheSame = true;

    printHeader();
    makeCsvHeader(csvFile);

    // Warm Up
    for(int i=0; i<20; ++i)
        performOldReduce(n, m, N, d_in, h_data);

    for( ; n<=1000000; n*=10, m/=10) {
        for(int i=0; i<ntest; ++i) {
            // Test Previous Redcution
            int nthreads = getLocalThreads(n);
            performTestLocal(n, m, N, d_in, d_in, h_data, nthreads, &record, csvFile);
        }        

        for(int optNumOperationPerThread = 32; optNumOperationPerThread<=4096; optNumOperationPerThread*=2) 
        {
            for(int i=0; i<ntest; ++i) {
                // Test Reduction 1.9
                performTestMany(n, m, N, d_in, d_in, h_data, &record, optNumOperationPerThread, csvFile);
            }
        }
    }
    
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_data);
    csvFile.close();
    return 0;
}

int getLocalThreads(int n) {
    if(n <= 64)  return 64;
    if(n <= 128) return 128;
    if(n <= 256) return 256;
    return 512;
}
void getArgs(int *ntest, char* deviceName, int argc, char** argv) {
    if(argc != 3) {
        std::cout << "Syntax: ./prog ntest deviceName\n";
        exit(1);
    }
    *ntest = atoi(argv[1]);
    strcpy(deviceName, argv[2]);
}

void printHeader() {
    std::cout << "---------- Reduction Test -----------" << std::endl;
}

void printResults(int n, int m, double *h_data) {
    int min = n < 3 ? n : 3; 
    for(int i=0; i<min; ++i) {            
        if (h_data[i] == m)
            std::cout << "Correct result: " << h_data[i] << std::endl;
        else 
            std::cout << "Incorrect result: " << h_data[i] << std::endl;
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

int sumOneArray(int m, double* d_in, double* d_out, int nthreads) {
    int err;
    if((err=local_reductionR(d_in, m, d_out, nthreads, 0))!=0) {
        return err;
    }
    return 0;
}

int performTestLocal(int n, int m, int N, double* d_in, double* d_out, double* h_data, int nthreads, Record* record, ofstream& csvFile) {
    record->algorithmName = algorithmLocal;
    record->numberOfArrays = n;
    record->numberOfElementsInArray = m;
    record->numberOfThreadInBlock = nthreads;
    record->numberOfBlocksPerArray = getBlockNum(m, nthreads);
    record->optNumOperationPerThread = 2;

    prepeateData(N, h_data);
    Measure_HtoD_time(N, d_in, h_data, record);
    Measure_Local(n, m, d_in, d_out, nthreads, record);
    Measure_DtoH_time(N, d_out, h_data, record);
    GetResultsForLocalReductionTo(record, h_data, n, m);
    writeRecord(csvFile, record);
    return 0;
}

int performTestManyOld(int n, int m, int N, double* d_in, double* d_out, double* h_data, Record* record, ofstream& csvFile) {
    record->algorithmName = algorithmMany1;
    record->numberOfArrays = n;
    record->numberOfElementsInArray = m;
    record->numberOfThreadInBlock = 512; //NTHREADS_MANY_OLD;
    record->numberOfBlocksPerArray = getBlockNum_Many_Old(m, record->numberOfThreadInBlock);
    record->optNumOperationPerThread = 2;
    
    prepeateData(N, h_data);
    Measure_HtoD_time(N, d_in, h_data, record);
    Measure_ManyOld(n, m, d_in, d_out, record);
    Measure_DtoH_time(N, d_out, h_data, record);
    GetResultsTo(record, h_data, n);
    writeRecord(csvFile, record);
    return 0;
}

int performTestMany(int n, int m, int N, double* d_in, double* d_out, double* h_data, Record* record, int optNumOperationPerThread, ofstream& csvFile) {
    record->algorithmName = algorithmMany2;
    record->numberOfArrays = n;
    record->numberOfElementsInArray = m;
    record->numberOfThreadInBlock = -1;
    record->numberOfBlocksPerArray = (int)ceil((float)m/(record->numberOfThreadInBlock));
    record->optNumOperationPerThread = optNumOperationPerThread;
    
    prepeateData(N, h_data);
    Measure_HtoD_time(N, d_in, h_data, record);
    Measure_Many(n, m, d_in, d_out, record, optNumOperationPerThread);
    Measure_DtoH_time(N, d_out, h_data, record);
    GetResultsTo(record, h_data, n);
    writeRecord(csvFile, record);
    return 0;
}

int performOldReduce(int n, int m, int N, double* d_in, double* h_data) {
    for (int i=0; i<N; ++i) 
        h_data[i] = 1;

    cudaMemcpy(d_in, h_data, N*sizeof(double), cudaMemcpyHostToDevice);

    int err;
    if((err=local_reductions_many_old(n, m, d_in, d_in))!=0) {
        return err;
    }

    return 0;
}

void prepeateData(int N, double* h_data) {
    for (int i=0; i<N; ++i) 
        h_data[i] = number;
}
void Measure_HtoD_time(int N, double* d_in, double* h_data, Record* record) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_in, h_data, N*sizeof(double), cudaMemcpyHostToDevice);
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    record->HtoDtime = duration;
}
void Measure_DtoH_time(int N, double* d_out, double* h_data, Record* record) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_data, d_out, N*sizeof(double), cudaMemcpyDeviceToHost);
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    record->DtoHtime = duration;
}
void Measure_ManyOld(int n, int m, double* d_in, double* d_out, Record* record) {
    auto start = std::chrono::high_resolution_clock::now();
    local_reductions_many_old(n, m, d_in, d_out);
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    record->kernelTime = duration;
}
void Measure_Local(int n, int m, double* d_in, double* d_out, int nthreads, Record* record) {
    size_t shift = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<n; ++i) {
        sumOneArray(m, d_in+shift, d_out+shift, nthreads);
        shift += m;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    record->kernelTime = duration;
}
void Measure_Many(int n, int m, double* d_in, double* d_out, Record* record, int optNumOperationPerThread) {
    auto start = std::chrono::high_resolution_clock::now();
    local_reductions_many(n, m, d_in, d_out, optNumOperationPerThread);    
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    record->kernelTime = duration;
}
void GetResultsTo(Record* record, double* h_data, int n) {
    record->result1 = h_data[0];
}

void GetResultsForLocalReductionTo(Record* record, double* h_data, int n, int m) {
    record->result1 = h_data[0];
}