#ifndef HNGPU_H
#define HNGPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"
#include <cub/cub.cuh>

#include <fstream>

#include "Environment.h"
#include "ccsr.h"
#include <iomanip>
#include <sstream>

__global__ struct __align__(8) HnFunction {
	int32_t mapping;
	uint32_t previous;
};

__global__ struct HnSolution {
	std::string * lhs, * rhs;
};

/* Solution Header Usage
* solutionHeader[0] = Function Offset
* solutionHeader[1] = Function Count
* solutionHeader[2] = Instance Size
* solutionHeader[3] = Thread Count
* solutionHeader[4] = New Function Offset
* solutionHeader[5] = New Function Count
*/

#define HNFUNC_OFFSET 0
#define HNFUNC_COUNT 1
#define HNFUNC_INSTSIZE 2
#define HNFUNC_THREADCOUNT 3
#define HNFUNC_NEWOFFSET 4
#define HNFUNC_NEWCOUNT 5

/* Requirement Header Usage
* requirementHeader[0] = Number Of Requirements
* requirementHeader[1] = Requirements Index
* requirementHeader[2] = First Query Vertex Depth 
*/

#define HNREQS_H_NUM 0
#define HNREQS_H_INDEX 1
#define HNREQS_H_FIRSTQVERTEX 2

/*
* Requirements Usage
* requirements[0 + 2*i] = Requirement Distance
* requirements[1 + 2*i] = Required Relation
*/

#define HNREQS_DIST 0
#define HNREQS_REL 1

namespace HnSetup {
	__host__ std::vector<HnSolution>* solve(const CCSR::CCSRGraph& ccsrQuery, const CCSR::CCSRGraph& ccsrData,
								const CCSR::CCSRStagger& queryStagger, const CCSR::CCSRStagger& dataStagger);

	__host__ uint32_t preProcessQuery(CCSR::CCSRGraph query, uint32_t** requirements, uint32_t** requirementHeader, int32_t** mappingData);

	__host__ void preinit(size_t MEM_LIMIT, size_t SOLN_LIMIT, size_t SCAN_LIMIT);

	__global__ void dummy();

	__host__ void gpuLaunch(const CCSR::CCSRGraph& query, const CCSR::CCSRGraph& data, const CCSR::CCSRStagger& dataStagger,
			uint32_t* requirements, uint32_t* requirementHeader,
			uint32_t maxValencyQuery, uint32_t maxValencyData, int32_t* mappingData,
			HnFunction** solutionCPU, uint32_t** solutionHeaderCPU,
			size_t MEM_LIMIT, size_t SOLN_LIMIT, size_t SCAN_LIMIT);

	__host__ void printFunctions(uint32_t* functionInfo, uint32_t width, uint32_t count);

	__host__ uint32_t functionsToFile(uint32_t* functionInfo, uint32_t width, uint32_t count);
	__host__ uint32_t functionsToFilePretty(uint32_t* functionInfo, uint32_t width, uint32_t count);
}

#endif