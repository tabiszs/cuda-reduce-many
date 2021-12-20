#ifndef TESTS_H
#define TESTS_H

using namespace std;

typedef struct Record {
    string deviceName;
    string programVersion;
    string algorithmName;
    int numberOfArrays;
    int numberOfElementsInArray;
    string dataType;
    double numberInCell;
    long HtoDtime;
    long kernelTime;
    long DtoHtime;
    int numberOfThreadInBlock;
    int numberOfBlocksPerArray;
    int optNumOperationPerThread;
    bool inOutTheSame;
    double result1;
    double result2;
    double result3;
} Record;

string getFileName(string deviceName);
void writeRecord(ofstream& file, Record* record);
string makeCsvRecord(string record);
void makeCsvHeader(ofstream& file);

#endif // TESTS_H
