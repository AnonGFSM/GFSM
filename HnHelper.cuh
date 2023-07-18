#ifndef HNHELPER_H
#define HNHELPER_H

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <algorithm>
#include "ccsr.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"
#include <cub/cub.cuh>

#define ERR_MISSING_FILE 232
#define ERR_INVALID_DATA 233
#define ERR_MISSING_DATA 234
#define ERR_NONLINEAR_DATA 235

struct OrderPair {
	uint32_t mapping;
	uint32_t size;
};

inline bool operator<(const OrderPair& pair0, const OrderPair& pair1) {
	if (pair0.size == pair1.size) {
		return pair0.mapping < pair1.mapping;
	}
	return pair0.size > pair1.size;
}


class Dynamic2D {

/*
* Structure:
	DynamicRow :: {Length, <Data>, location of more data}
*	FrontRows: {DynamicRow0, DynamicRow1, DynamicRow2, ...}
*	BackRows: Data for longer rows
*/


private:
	uint32_t* frontRows, * backRows;
	uint32_t rows, width, reserved;

	uint32_t bAllocated;

	void construct(uint32_t _rows, uint32_t _width, uint32_t _reserve) {
		rows = _rows;
		width = _width;
		reserved = _reserve * (width + 2);
		bAllocated = width + 2;
		frontRows = (uint32_t*)calloc(rows * (width + 2), sizeof(uint32_t));
		backRows = (uint32_t*)calloc(reserved, sizeof(uint32_t));
	}

public:

	uint32_t get(uint32_t* alloc, int i) const {
		if(width <= i){
			uint32_t offset = *(alloc + width + 1);
			if (offset) {
				return get(&backRows[offset], i - width);
			} 
			return -1; //Debug State
		}
		return alloc[i+1];
	}

	uint32_t* getBottomRow(uint32_t* alloc) {
		if (alloc[0] >= width) {
			uint32_t offset = *(alloc + width + 1);
			if (offset) {
				return getBottomRow(&backRows[offset]);
			}

			*(alloc + width + 1) = bAllocated;
			uint32_t* result = &backRows[bAllocated];
			bAllocated += width + 2;

			return result;
		}
		return alloc;
	}

	void put(uint32_t* alloc, uint32_t value) {
		uint32_t* row = getBottomRow(alloc);
		row[row[0] + 1] = value;
		row[0]++;
		if (row != alloc) {
			alloc[0]++;
		}
	}
	
struct DynamicRow{
	uint32_t* allocation;
	uint32_t width, size;
	Dynamic2D* parent;

	uint32_t operator [](int i) const {
		if (width <= i) {
			return parent->get(allocation, i);
		}
		return allocation[i+1];
	}

	void operator << (uint32_t value) {
		size++;
		parent->put(allocation, value);
	}
};

	Dynamic2D(uint32_t _rows, uint32_t _width, uint32_t reserved) {
		construct(_rows, _width, reserved);
	}

	Dynamic2D(uint32_t _rows, uint32_t _width) {
		construct(_rows, _width, _rows);
	}

	~Dynamic2D() {
		free(frontRows);
		free(backRows);
	}

	DynamicRow operator [](int i) const{
		uint32_t* row = &frontRows[i * (width+2)];
		return { row, width, row[0], (Dynamic2D*) this };
	}

	void copy(uint32_t* target, DynamicRow row) {
		if (width < row.size) {
			memcpy(target, row.allocation + 1, width * sizeof(uint32_t));
			copy(target + width, { &backRows[*(row.allocation + width + 1)], width, row.size - width, (Dynamic2D*)this });
		}
		else {
			memcpy(target, row.allocation+1, row.size * sizeof(uint32_t));
		}
	}

//#define DEBUGDR
#ifdef DEBUGDR
	void dumpBfr() {
		printf("\nFront Buffer:");
		for (uint32_t i = 0; i < (width+2) * rows; i++) {
			if (!(i % (width + 2))) {
				printf("\n");
			}
			printf("%lu, ", frontRows[i]);
		}

		printf("\nBack Buffer:");
		for (uint32_t i = 0; i < reserved; i++) {
			if (!(i % (width + 2))) {
				printf("\n");
			}
			printf("%lu, ", backRows[i]);
		}
	}
#endif
};


enum ParseMode { header_p, vertex_p, edge_p };
enum ParseState { complete_p, fail_p, edges_found_p, end_p };

#define PARSE_PROTECT //Who knows what people will throw at it.

void clearValues(uint32_t* values, size_t size) {
	memset(values, 0, size*sizeof(uint32_t));
}

inline ParseState parseValues(char** str, char* end, uint32_t* values, ParseMode pm){

	uint32_t scanWidth;
	char ch = **str;

	if (ch == 't') {
		if (strncmp(*str, "t # -1", 6)) {
			return fail_p;
		}
		return end_p;
	}

	switch(pm){
		case header_p:
			scanWidth = 4;
			(*str)-=2; //This is bad, but lets me keep my for loop simple

			break;

		case vertex_p:
			if(ch == 'e'){
				return edges_found_p;
			}
			if(ch != 'v'){
				return fail_p;
			}

			scanWidth = 2;
			break;

		case edge_p:
			if(ch != 'e'){
				return fail_p;
			}

			scanWidth = 3;
			break;

	}

	uint32_t currentValue = 0;
	for ((*str)+=2; *str < end; (*str)++) {
		ch = **str;

		if (ch >= 48 && ch <= 57) {
			uint32_t i = ch - 48;
			values[currentValue] *= 10;
			values[currentValue] += i;
		}
		else {
			currentValue++;

			if (ch == '\n') {

#ifdef PARSE_PROTECT
				if(currentValue < scanWidth){
					return fail_p;
				}
#endif
				(*str)++;
				break;
			}

#ifdef PARSE_PROTECT
			if (currentValue > scanWidth || ch != ' ') {
				return fail_p;
			}
#endif

		}
	}

	return complete_p;
}

//30264781
void dumpBfr(uint32_t* bfr, uint32_t size) {
	printf("\Buffer Dump:");
	for (uint32_t i = 0; i < size; i++) {
		if (!(i % 10)) {
			printf("\n");
		}
		printf("%i, ", bfr[i] & CCSRIndex(CCSRVertSize));
	}
}


CCSR::CCSRGraph txtGSI(std::string loc, bool directed, CCSR::CCSRStagger* stagger) {

	auto start = std::chrono::steady_clock::now();
	auto fstart = std::chrono::steady_clock::now();
	FILE* f = fopen(loc.c_str(), "r");

	if (f == NULL) {


		std::cout << "\nMissing file " << loc << std::endl;
		exit(ERR_MISSING_FILE);
	}

	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	rewind(f);


	char* buf = (char*)calloc(size, sizeof(char));

	fread(buf, sizeof(char), size, f);
	auto fend = std::chrono::steady_clock::now();
	std::chrono::duration<double> felapsed_seconds = fend - fstart;
	info_printf("\nFile Access Time: %fs\n", felapsed_seconds.count());

	char* strEnd = buf + size * sizeof(char);

	char* strPtr = buf;

	if (strncmp(buf, "t # 0\n", 6)) {
		exit(ERR_INVALID_DATA); //BROKE
	}

	strPtr += 6;

	uint32_t values[4]{};
	ParseState pState;
	//Parse Header

	pState = parseValues(&strPtr, strEnd, values, header_p);
	if (pState == fail_p) {
		exit(ERR_INVALID_DATA);
	}

	uint32_t verticeCount = values[0];
	uint32_t edgeCount = values[1];
	uint32_t labelCount = values[2];
	//Throwaway values[3] as we don't use labelled vertices

	//Scan Vertex Data
	uint32_t vertexSeen = 0;

	for (pState = complete_p; pState == complete_p; ) {
		clearValues(values, 2);
		pState = parseValues(&strPtr, strEnd, values, vertex_p);
		if (pState == fail_p) {
			exit(ERR_INVALID_DATA);
		}
		if (pState == end_p) {
			break;
		}
		vertexSeen++;
	}

	if (vertexSeen < verticeCount) {
		exit(ERR_MISSING_DATA);
	}

	
	OrderPair* orderings = (OrderPair*)malloc(sizeof(OrderPair) * verticeCount);

	uint32_t preSize = ((edgeCount / verticeCount) + 1);

	auto start2 = std::chrono::steady_clock::now();
	Dynamic2D allocations(verticeCount, preSize, verticeCount);
	auto end2 = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
	info_printf("\nAlloc Time: %f s\n", elapsed_seconds2.count());

	auto startV = std::chrono::steady_clock::now();

	uint32_t edgeSeen = 0;
	for (pState = complete_p; pState == complete_p; edgeSeen++) {
		clearValues(values, 3);
		pState = parseValues(&strPtr, strEnd, values, edge_p);
		if (pState == fail_p) {
			exit(ERR_INVALID_DATA);
		}
		if (pState == end_p) {
			break;
		}

		constexpr auto vertMask = CCSRIndex(CCSRVertSize);
		allocations[values[0]] << ((values[1] & vertMask) | (values[2] << CCSRVertSize));
	}

	if (edgeSeen < edgeCount) {
		exit(ERR_MISSING_DATA);
	}

	uint32_t allocationSize = 0;
	uint32_t maxLen = 0;
	for (uint32_t i = 0; i < verticeCount; i++) {
		auto row = allocations[i];
		if (row.size) {
			allocationSize += row.size + 1;
		}

		maxLen = std::max(maxLen, row.size);
		orderings[i] = { i, (uint32_t)row.size };
	}

	CCSRSegment* ccsrData = (CCSRSegment*)malloc(sizeof(CCSRSegment) * allocationSize);
	uint32_t* verticeLocs = new uint32_t[verticeCount]{};
	uint32_t* inverseVerticeLocs = new uint32_t[verticeCount]{};

	CCSRSegment* ccsrWrite = ccsrData;

	uint32_t* staggers = new uint32_t[maxLen]{};
	CCSR::StaggerInfo* staggersInfo = new CCSR::StaggerInfo[maxLen]{};
	uint32_t* degreeCounts = new uint32_t[maxLen]{};

	uint32_t vSize;

	auto endV = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_secondsV = endV - startV;
	info_printf("\nParse - First View Time: %fs\n", elapsed_secondsV.count());

	auto startS = std::chrono::steady_clock::now();

	std::sort(orderings, orderings + verticeCount);

	uint32_t actualSize = 0;

	for (uint32_t i = 0; i < verticeCount; i++) {
		uint32_t currentVertex = orderings[i].mapping;
		if (currentVertex < 0 || currentVertex >= verticeCount) {
			exit(1);
		}
		auto row = allocations[currentVertex];

		vSize = row.size;
		verticeLocs[currentVertex] = (uint32_t) (ccsrWrite - ccsrData);
		inverseVerticeLocs[i] = currentVertex;
		if (vSize != 0) {
			*ccsrWrite = vSize;
			allocations.copy(ccsrWrite + 1, row);
			staggers[vSize-1] += vSize + 1;
			degreeCounts[vSize - 1]++;
			ccsrWrite += vSize + 1;
			actualSize++;
		}

	}

	auto endS = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_secondsS = endS - startS;
	info_printf("\nParse - Data Access Time: %fs\n", elapsed_secondsS.count());

	//Assuming top heavy CCSR
	staggersInfo[maxLen - 1].offset = 0;
	staggersInfo[maxLen - 1].normOffset = 0;
	for (int32_t i = maxLen - 2; i >= 0; i--) {
		staggersInfo[i].offset = staggersInfo[i + 1].offset + staggers[i + 1];
		staggersInfo[i].normOffset = staggersInfo[i + 1].normOffset + degreeCounts[i + 1];
	}

	auto startM = std::chrono::steady_clock::now();

	for (uint32_t i = 0; i < allocationSize; i++) {
		uint32_t value = ccsrData[i];
		uint32_t rel = value & CCSRRelation(CCSRRelSize);
		if (rel) { //TODO: Gonna assume that valency never greater than vertex size, probs should put a guard somewhere
			constexpr auto vertMask = CCSRIndex(CCSRVertSize);
			uint32_t vertex = verticeLocs[value & vertMask];
			ccsrData[i] = vertex | rel;
		}
	}

	auto endM = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_secondsM = endM - startM;
	info_printf("\nParse - Remap Time: %fs\n", elapsed_secondsM.count());

	auto startSo = std::chrono::steady_clock::now();
	CCSRSegment* temp = new CCSRSegment[maxLen];

	uint32_t len = 0;
	for (uint32_t i = 0; i < allocationSize; i += len + 1) {
		if (len = ccsrData[i]) {
			CCSRSegment* row = &ccsrData[i] + 1;

			//TODO: Safety we can enable but is slower!
			//CCSR::insNodeSort(row, row + len, temp);
			if (!std::is_sorted(row, row + len, CCSR::nodeCompare)) {
				std::sort(row, row + len, CCSR::nodeCompare);
			}
		}
	}

	auto endSo = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_secondsSo = endSo - startSo;
	info_printf("\nParse - Resort Time: %fs\n", elapsed_secondsSo.count());

	delete[] temp;

	stagger->size = maxLen;
	stagger->staggers = staggers;
	stagger->staggersInfo = staggersInfo;


	CCSR::CCSRGraph graph{ ccsrData, verticeLocs, inverseVerticeLocs, verticeCount, allocationSize };

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	info_printf("\nOverall Parse Time: %fs\n", elapsed_seconds.count());
	csv_printf("%fs,", elapsed_seconds.count());

	return graph;
}

#endif