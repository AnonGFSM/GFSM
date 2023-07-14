#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "HnHelper.cuh"
#include "HnGPU.cuh"

#include <chrono>
#include <iostream>


__global__ void dummyWarm() {

}

void argTest(std::string query, std::string data) {
    int* test;
    cudaDeviceReset();

    for (int i = 0; i < 5; i++) {
        cudaMalloc(&test, UPPERMEMLIMIT);
        dummyWarm <<<1, 1 >>> ();
        cudaFree(test);
    }

#ifdef NAMEDATA
    std::cout << "\n\n----------------------------------------------------------------------\n\n";
    std::cout << "       " << "GSI: " << query << " --> " << data;
    std::cout << "\n\n----------------------------------------------------------------------\n\n";
#endif
    //This is bad!!
    csv_printf("\n%s,%s,", query.substr(query.find_last_of("/") + 1).c_str(), data.substr(data.find_last_of("/")+1).c_str());

    HnSetup::preinit(MEMLIMIT, MAXSOLNSIZE, MAXSCANSIZE);
    auto start = std::chrono::steady_clock::now();

    CCSR::CCSRStagger queryStagger;
    CCSR::CCSRGraph queryGraph = txtGSI(query, false, &queryStagger);
    //print(queryGraph);

    CCSR::CCSRStagger dataStagger;
    CCSR::CCSRGraph dataGraph = txtGSI(data, false, &dataStagger);
    //print(dataGraph);

#ifdef FINGERPRINTING
    //Hashing Time!!
    SHA256 shaQuery;
    shaQuery.update((uint8_t*) queryGraph.segments, queryGraph.size * sizeof(uint32_t));
    uint8_t* digestQ = shaQuery.digest();
    debug_printf("\nQuery Graph, SHA256: %s", SHA256::toString(digestQ).c_str());
    delete[] digestQ;

    SHA256 shaData;
    shaData.update((uint8_t*)dataGraph.segments, dataGraph.size * sizeof(uint32_t));
    uint8_t* digestD = shaData.digest();
    debug_printf("\nData Graph, SHA256: %s", SHA256::toString(digestD).c_str());
    delete[] digestD;
#endif

    HnSetup::solve(queryGraph, dataGraph, queryStagger, dataStagger);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    info_printf("\nEnd to End Time: %fs\n", elapsed_seconds.count());
    csv_printf("%fs,", elapsed_seconds.count());
}

void nullArgTest() {
    //argTest("query/_5_road5.g", "data/roadNet-PA.g"); //"data/roadNet-PA.g"
    //argTest("query/_3_triangle.g", "data/cit-Patents-Sorted.g"); //"data/roadNet-PA.g"
    //argTest("query/DBpedia-Queries/_2_query_1.g", "data/DBPedia.g"); //"data/roadNet-PA.g"
    argTest("query/_3_triangle.g", "data/roadNet-PA.g");
}

//__global__ void dummyKernel() {

//}

__global__ void TestKernel(int* test) {
    printf("Test %i", test[100000]);
}


int main(int argc, char* argv[])
{
    if (argc == 3) {
        argTest(std::string(argv[1]), std::string(argv[2]));
    }
    else {
        nullArgTest();
    }

}
