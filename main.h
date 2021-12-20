#ifndef MAIN_H
#define MAIN_H

#include "tests.h"

int getLocalThreads(int n);
void getArgs(int *ntest, char* deviceName, int argc, char** argv);
void printHeader();
void printResults(int n, int m, double *h_data);
void printResultsForLocalReduction(int n, int m, double *h_data);

int sumOneArray(int m, double* d_in, double* d_out, int nthreads);
int performTestLocal(int n, int m, int N, double* d_in, double* d_out, double* h_data, int nthreads, Record* record, ofstream& csvFile);
int performTestManyOld(int n, int m, int N, double* d_in, double* d_out, double* h_data, Record* record, ofstream& csvFile);
int performTestMany(int n, int m, int N, double* d_in, double* d_out, double* h_data, Record* record, int optNumOperationPerThread, ofstream& csvFile);
int performOldReduce(int n, int m, int N, double* d_in, double* h_data);

void prepeateData(int N, double* h_data);
void Measure_HtoD_time(int N, double* d_in, double* h_data, Record* record);
void Measure_DtoH_time(int N, double* d_out, double* h_data, Record* record);
void Measure_ManyOld(int n, int m, double* d_in, double* d_out, Record* record);
void Measure_Local(int n, int m, double* d_in, double* d_out, int nthreads, Record* record);
void Measure_Many(int n, int m, double* d_in, double* d_out, Record* record, int optNumOperationPerThread);
void GetResultsTo(Record* record, double* h_data, int n);
void GetResultsForLocalReductionTo(Record* record, double* h_data, int n, int m);

#endif