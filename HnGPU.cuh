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


//__global__ struct __align__(8) MappingPair {
//	uint32_t lhs, rhs;
//};

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
	//Segment it to make timing more readable
	__host__ void gpuLaunch(const CCSR::CCSRGraph& query, const CCSR::CCSRGraph& data, const CCSR::CCSRStagger& dataStagger,
			uint32_t* requirements, uint32_t* requirementHeader,
			uint32_t maxValencyQuery, uint32_t maxValencyData, int32_t* mappingData,
			HnFunction** solutionCPU, uint32_t** solutionHeaderCPU,
			size_t MEM_LIMIT, size_t SOLN_LIMIT, size_t SCAN_LIMIT);

	__host__ void printFunctions(uint32_t* functionInfo, uint32_t width, uint32_t count);

	__host__ uint32_t functionsToFile(uint32_t* functionInfo, uint32_t width, uint32_t count);
	__host__ uint32_t functionsToFilePretty(uint32_t* functionInfo, uint32_t width, uint32_t count);
}

	class SHA256 {

	public:
		SHA256();
		void update(const uint8_t* data, size_t length);
		void update(const std::string& data);
		uint8_t* digest();

		static std::string toString(const uint8_t* digest);

	private:
		uint8_t  m_data[64];
		uint32_t m_blocklen;
		uint64_t m_bitlen;
		uint32_t m_state[8]; //A, B, C, D, E, F, G, H

		static uint32_t rotr(uint32_t x, uint32_t n);
		static uint32_t choose(uint32_t e, uint32_t f, uint32_t g);
		static uint32_t majority(uint32_t a, uint32_t b, uint32_t c);
		static uint32_t sig0(uint32_t x);
		static uint32_t sig1(uint32_t x);
		void transform();
		void pad();
		void revert(uint8_t* hash);
	};

	std::string checkSumCuda(void* cudaBfr, size_t size);

#endif