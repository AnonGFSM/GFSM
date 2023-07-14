#pragma once
#include "ccsr.h"
#include <bitset>
#include <algorithm>


namespace CCSR {

	bool nodeCompare(CCSRSegment seg0, CCSRSegment seg1) {
		constexpr auto mask = CCSRIndex(CCSRVertSize);
		return (seg0 & mask) < (seg1 & mask);
	}

	inline bool inNodeCompare(CCSRSegment seg0, CCSRSegment seg1) {
		constexpr auto mask = CCSRIndex(CCSRVertSize);
		return (seg0 & mask) < (seg1 & mask);
	}

	//STL Variance Means I've Reimplemented this!!
	inline CCSRSegment* nodeUpperBound(CCSRSegment* begin, CCSRSegment* end, CCSRSegment value) {
		for (auto it = begin; it != end; it++) {
			if (inNodeCompare(value, *it)) {
				return it;
			}
		}
		return end;
	}

	inline void rotateNodes(CCSRSegment* begin, CCSRSegment* n_begin, CCSRSegment* end, CCSRSegment* temp) {
		int dist = n_begin - begin;
		int len = end - begin;
		for (int i = 0; i < len; i++) {
			temp[(abs(i - dist)) % len] = begin[i];
		}
		for (int i = 0; i < len; i++) {
			begin[i] = temp[i];
		}
		//memcpy(begin, temp, len * sizeof(CCSRSegment));
	}


	void printSegs(CCSRSegment* begin, CCSRSegment* end) {
		printf("\nPrinting Seg Row: ");
		for (auto it = begin; it < end; it++) {
			printf("%i, ", *it);
		}
	}

	inline bool ordered(CCSRSegment* begin, CCSRSegment* end) {
		for (auto it = begin+1; it != end; it++)
		{
			if (inNodeCompare(*it, *(it-1))) {
				return false;
			}
		}
		return true;
	}

	void insNodeSort(CCSRSegment* begin, CCSRSegment* end, CCSRSegment* temp)
	{
		if (ordered(begin, end)) {
			return;
		}

		for (auto it = begin; it != end; it++)
		{
			auto const insertion_point = nodeUpperBound(begin, it, *it);
			//auto const insertion_point = std::upper_bound(begin, it, *it, nodeCompare);

			// Shifting the unsorted part
			//std::rotate(insertion_point, it, it + 1);
			rotateNodes(insertion_point, it, it + 1, temp);
		}
	}

//Horrible Unwrapped Badness

/*
#define cCHECK(l, r) nodeCompare(begin[l], begin[r])
#define initArray(i0, i1, i2, i3) CCSRSegment arr [] = { begin[i0], begin[i1], begin[i2], begin[i3] }; memcpy(begin, arr, sizeof(arr));

	void nodeSort(CCSRSegment* begin, CCSRSegment* end) {
		switch (int len = end - begin) {
			case 4:
				if (cCHECK(0, 1)) {
					if (cCHECK(1, 2)) {
						if (cCHECK(2, 3)) {
							initArray(0, 1, 2, 3);
						}
						else if (cNodeCompare(begin + 1, begin + 3, len, 4, nodeCompare)) {

						}
						else if (cNodeCompare(begin, begin + 2, len, 4, nodeCompare)) {

						}
						else {

						}
					} 
					else if (cNodeCompare(begin, begin + 2, len, 4, nodeCompare)) {

					}
					else {

					}
				}
				else {
					if (cNodeCompare(begin, begin + 2, len, 4, nodeCompare)) {

					}
					else if (cNodeCompare(begin + 1, begin + 2, len, 4, nodeCompare)) {

					}
					else {

					}
				}
				return;
			case 3:
			case 2:
			case 1:
		}
		std::sort(begin, end, nodeCompare);
	}
*/
	bool intDesc(int i, int i2) {
		return i > i2;
	}

	inline uint32_t getIndex(uint32_t* arr, uint32_t len, uint32_t value) {
		return std::find(arr, arr+len, value) - arr;
	}

	struct ReqRelPair {
		uint32_t req, rel;
	};

	bool reqRelCompare(ReqRelPair l, ReqRelPair r) {
		return (l.req) < (r.req);
	}

	//Formatting of MappingData
	//-1 means Vertex unseen
	// 0 means Vertex seen   and not mapped
	//>0 means Vertex seen   and mapped

	void genereateCCSRRequirements(CCSRGraph graph, uint32_t* requirements, uint32_t* requirementHeader,
		uint32_t maxValency, uint32_t depth,
		int32_t* mappingData) {

		uint32_t numOfReqs = 0;
		CCSRSegment* localVerticeLocs = new CCSRSegment[graph.count]{};
		uint32_t offset = 0;
		for (uint32_t i = 0; i < graph.count; i++) {
			localVerticeLocs[i] = offset;
			offset += graph.segments[offset] + 1;
		}

		//graph.verticeLocs[depth];

		//printf("\nOld: ");
		//for (int i = 0; i < graph.count; i++) {
		//	printf("%i, ", mappingData[i]);
		//}

		uint32_t relativeDepth;
		bool found = false;

		//Find a vertex to map for.
		for (uint32_t i = 0; i < graph.count; i++) {
			if (mappingData[i] == 0 && (!found)) {
				mappingData[i] = 1;
				relativeDepth = i;
				found = true;
			} else if (mappingData[i] > 0) {
				mappingData[i]++;
			}
			
		}

		//printf("\nNew: ");
		//for (int i = 0; i < graph.count; i++) {
		//	printf("%i, ", mappingData[i]);
		//}

		if (!found) {
		//	printf("\n Error: Could not fully traverse Query!");
			exit(1);
		}

		CCSRSegment* examinedRow = &graph.segments[localVerticeLocs[relativeDepth]];
		CCSRSegment* currentRow = examinedRow + 1;
		uint32_t examinedIndex = examinedRow - graph.segments;

		uint32_t reqOffset = depth * maxValency * 2;
		uint32_t firstQ = 0;

		ReqRelPair* reqRels = new ReqRelPair[maxValency]();

		uint32_t currentVertex, currentIndex;


		for (int i = 0; i < *examinedRow; i++, currentRow++) {
			const auto vertMask = CCSRIndex(CCSRVertSize);
			const auto relMask = CCSRRelation(CCSRRelSize);
			currentVertex = (*currentRow) & vertMask;
			currentIndex = getIndex(localVerticeLocs, graph.count, currentVertex); //THIS IS BROKEN AND WRONG!!!!!!!!!!!!

			if (mappingData[currentIndex] < 0) {
				mappingData[currentIndex] = 0;
			}

			if (mappingData[currentIndex] > 0) {
				uint32_t relation = (*currentRow) & relMask;
				CCSRSegment* reflectedRow = &graph.segments[(*currentRow) & vertMask];
				for (int i2 = 0; i2 < *reflectedRow; i2++) {
					if ((*(reflectedRow + 1 + i2) & vertMask) == localVerticeLocs[relativeDepth]) {
						relation = (*(reflectedRow + 1 + i2)) & relMask;
						break;
					}
				}

				reqRels[numOfReqs].req = (uint32_t) (mappingData[currentIndex]-1);
				reqRels[numOfReqs].rel = relation;
				numOfReqs++;
			}

		}


		//printf("\nTest Data");

		//for (int i = 0; i < numOfReqs; i++) {
		//	printf("(%u, %u),", reqs[i], rels[i]);
		//}
	
		if (numOfReqs) {
			std::sort(reqRels, reqRels + numOfReqs, reqRelCompare);

			//printf("\nReq Order: ");
			//for (int i = 0; i < numOfReqs; i++) {
			//	printf("%u, ", reqs[i]);
			//}

			requirements[reqOffset] = reqRels[0].req;
			requirements[reqOffset + 1] = reqRels[0].rel;

			for (int i = 1; i < numOfReqs; i++) {
				requirements[reqOffset + i * 2] = reqRels[i].req - reqRels[i-1].req;
				requirements[reqOffset + i * 2 + 1] = reqRels[i].rel;
			}

			firstQ = requirements[reqOffset];
			for (int i = 0; i < numOfReqs; i++) {
				firstQ = requirements[reqOffset + i*2];
				if (firstQ) {
					break;
				}
			}
		}
		else {
			firstQ = 0;
		}

		if ((firstQ > depth && depth) || (depth && !firstQ)) {
			debug_printf("\nQuery Process Failure: Depth %u, First Q %u", depth, firstQ);
			csv_printf("Query Process Failure: Depth %u, First Q %u,", 0);
			exit(1);
		}

		delete[] reqRels;
		delete[] localVerticeLocs;

		requirementHeader[depth * 3 + 0] = numOfReqs;
		requirementHeader[depth * 3 + 1] = reqOffset;
		requirementHeader[depth * 3 + 2] = firstQ;
	}

	uint32_t getMaxValency(CCSRGraph graph) {

		uint32_t maxValency = 0;
		CCSRSegment* ceiling = graph.segments + graph.size;

		for (CCSRSegment* examinedRow = graph.segments; examinedRow < ceiling; examinedRow += (*examinedRow)+1){
			if (*examinedRow > maxValency) {
				maxValency = *examinedRow;
			}
		}

		return maxValency;

	}

	void check(CCSRGraph graph) {
		for (int i = 0; i < graph.size; i++) {
			if (graph.segments[i] && !(graph.segments[graph.segments[i] & CCSRIndex(CCSRVertSize)])) {
				printf("\nBAD, %i from %i", graph.segments[graph.segments[i] & CCSRIndex(CCSRVertSize)], graph.segments[i]);
			}
		}
	}

//#define SMALLPRINT
//#define RAWPRINT
//#define EXTRADATA

	void print(CCSRGraph graph) {
		std::bitset<32> relC(CCSRRelation(CCSRRelSize));
		std::bitset<32> indexC(CCSRIndex(CCSRVertSize));
		std::cout << "\nRelation Bits:" << relC;
		std::cout << "\nIndex Bits:" << indexC;

#ifdef RAWPRINT
		std::cout << "\nRaw:";
		for (int i = 0; i < graph.size; i++) {
			printf("%lu, ", graph.segments[i]);

#if defined(SMALLPRINT) && defined(RAWPRINT)
			if (i > 300) {
				printf("\n TOO MUCH PRINTED");
				break;
			}
#endif
		}
#endif

		uint32_t node = 0;
		for (uint32_t i = 0; i < graph.size; i++, node++) {

			uint32_t nodeDegree = graph.segments[i];
			printf("\nNode %u (%u), NodeDeg %u: ", node, i, nodeDegree);

			for (uint32_t  i2 = 0; i2 < nodeDegree; i2++, i++) {
				uint32_t nodeValue = graph.segments[i+1];
				printf("{R: %u (%u), V: %u} ", (nodeValue & CCSRRelation(CCSRRelSize)) >> CCSRVertSize, nodeValue & CCSRRelation(CCSRRelSize), nodeValue & CCSRIndex(CCSRVertSize));
			}

#ifdef SMALLPRINT
			if (i > 3000) {
				printf("\n TOO MUCH PRINTED");
				break;
			}
#endif
		}
#ifdef EXTRADATA
		printf("\nInverse-Vertice Locations\n");
		for (uint32_t i = 0; i < graph.count; i++) {
			printf("(%i -> %i)", i, graph.invVerticeLocs[i]);
		}

		printf("\nVertice Locations\n");
		for (uint32_t i = 0; i < graph.count; i++) {
			printf("(%i -> %i)", i, graph.verticeLocs[i]);
		}
#endif
	}

	void save(std::string name, CCSRGraph graph) {
		std::ofstream os("debugGraph.g", std::ofstream::out | std::ofstream::trunc);
		os << "CCSR Graph: " << name;
		for (uint32_t i = 0; i < graph.size; i++) {
			os << "\nVertex " << std::to_string(i) << ": ";
			uint32_t degree = graph.segments[i];
			for (uint32_t i2 = 1; i2 <= degree; i2++) {
				uint32_t segment = graph.segments[i + i2];
				os << "(" << std::to_string(segment & CCSRIndex(CCSRVertSize)) 
					<< ", " << std::to_string((segment & CCSRRelation(CCSRRelSize))>>CCSRVertSize) 
					<< "), ";
			}
			i += degree;
		}
		os.close();
	}
}