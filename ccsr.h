#ifndef CCSRH
#define CCSRH
#include "cuda_runtime.h"
#include "Environment.h"
#include <stdint.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#define CCSRSegment uint32_t

namespace CCSR {
	bool nodeCompare(CCSRSegment seg0, CCSRSegment seg1);
	void insNodeSort(CCSRSegment* begin, CCSRSegment* end, CCSRSegment* temp);
	void printSegs(CCSRSegment* begin, CCSRSegment* end);

	__global__ struct __align__(8) CSRSegment {
		uint32_t mapping;
		uint32_t relation;
	};

	__global__ struct CCSRGraph {
		CCSRSegment* segments;
		uint32_t* verticeLocs;
		uint32_t* invVerticeLocs;
		uint32_t count, size;
	};

	__global__ struct CCSRReqs {
		uint32_t* reqs;
		uint32_t* header;
	};

	__global__ struct __align__(8) StaggerInfo {
		uint32_t offset;
		uint32_t normOffset;
	};

	__host__ struct CCSRStagger {
		uint32_t* staggers;
		StaggerInfo* staggersInfo;
		uint32_t size;
	};

	void genereateCCSRRequirements(CCSRGraph graph, uint32_t* requirements, uint32_t* requirementHeader, uint32_t maxValency, uint32_t depth, int32_t* mappingData);
	uint32_t getMaxValency(CCSRGraph graph);
	void check(CCSRGraph graph);
	void print(CCSRGraph graph);
	void save(std::string name, CCSRGraph graph);
}

#endif