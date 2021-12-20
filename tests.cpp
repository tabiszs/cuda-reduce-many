// Author: Stanisław Tabisz
// Date: 25-11-2021

#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <ctime>
#include "tests.h"

using namespace std;

string getFileName(string deviceName) {
    std::time_t t = std::time(0);   // get time now
    std::tm* now = std::localtime(&t);
    string fileName;
    ostringstream nameStream = ostringstream();
    nameStream
        << deviceName << '_'
        << (now->tm_year + 1900) << '-'
        << (now->tm_mon + 1) << '-'
        << now->tm_mday << '_'
        << now->tm_hour << '.'
        << now->tm_min << ".csv";
    return nameStream.str();
}

void writeRecord(ofstream& file, Record* record) {
    file
        << makeCsvRecord(record->deviceName)
        << makeCsvRecord(record->programVersion)
        << makeCsvRecord(record->algorithmName)
        << makeCsvRecord(to_string(record->numberOfArrays))
        << makeCsvRecord(to_string(record->numberOfElementsInArray))
        << makeCsvRecord(record->dataType)
        << makeCsvRecord(to_string(record->numberInCell))
        << makeCsvRecord(to_string(record->HtoDtime))
        << makeCsvRecord(to_string(record->kernelTime))
        << makeCsvRecord(to_string(record->DtoHtime))
        << makeCsvRecord(to_string(record->numberOfThreadInBlock))
        << makeCsvRecord(to_string(record->numberOfBlocksPerArray))
        << makeCsvRecord(to_string(record->optNumOperationPerThread))
        << makeCsvRecord(to_string(record->inOutTheSame))
        << record->result1
        << std::endl;
}

string makeCsvRecord(string record) {
    return record + ";";
}

void makeCsvHeader(ofstream& file) {
    file
        << "Device;"
        << "Program version;"
        << " Algorithm name;"
        << "Number of Arrays;"
        << "Number of elements in array;"
        << "Type of data;"
        << "Number in cell;"
        << "HtoD time [ns];"
        << "Kernel time [ns];"
        << "DtoH time [ns];"
        << "Number of thread in block;"
        << "Number of thread in grid;"
        << "Optimum add operation per thread;"
        << "Input & output in the same point in memory;"
        << "Result1;"
        << "Result2;"
        << "Result3"
        << std::endl;
}
