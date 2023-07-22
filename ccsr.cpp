#pragma once
#include "ccsr.h"
#include <bitset>
#include <algorithm>


namespace CCSR {

	/*
	* Node Compare (is_sorted requires an non-inlined version)
	* Comparison Function of two CCSR elements
	*/
	bool nodeCompare(CCSRSegment seg0, CCSRSegment seg1) {
		constexpr auto mask = CCSRIndex(CCSRVertSize);
		return (seg0 & mask) < (seg1 & mask);
	}

	/*
	* Inline Node Compare
	* Comparison Function of two CCSR elements
	*/
	inline bool inNodeCompare(CCSRSegment seg0, CCSRSegment seg1) {
		constexpr auto mask = CCSRIndex(CCSRVertSize);
		return (seg0 & mask) < (seg1 & mask);
	}

	/*
	* Rotate Nodes
	* Flips a region of data
	*/
	inline void rotateNodes(CCSRSegment* begin, CCSRSegment* n_begin, CCSRSegment* end, CCSRSegment* temp) {
		int dist = n_begin - begin;
		int len = end - begin;
		for (int i = 0; i < len; i++) {
			temp[(abs(i - dist)) % len] = begin[i];
		}
		for (int i = 0; i < len; i++) {
			begin[i] = temp[i];
		}
	}

	/*
	* Pretty Prints CCSR elements
	*/
	void printSegs(CCSRSegment* begin, CCSRSegment* end) {
		printf("\nPrinting Seg Row: ");
		for (auto it = begin; it < end; it++) {
			printf("%i, ", *it);
		}
	}

	/*
	* Is ordered?
	* Checks whether a CCSR region is ordered
	*/
	inline bool ordered(CCSRSegment* begin, CCSRSegment* end) {
		for (auto it = begin+1; it != end; it++)
		{
			if (inNodeCompare(*it, *(it-1))) {
				return false;
			}
		}
		return true;
	}

	/*
	* Get Index
	* Find - Distance to beginning
	*/
	inline uint32_t getIndex(uint32_t* arr, uint32_t len, uint32_t value) {
		return std::find(arr, arr+len, value) - arr;
	}

	/*
	* Sorting Pair before converted to CCSR
	*/

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

	/*
	* Convert Query Graph into requirements (to solve subgraph isomorphism over)
	* Note: GFSM is unaware of the actual query at runtime (or even its nodes), and only 
	*		cares to enforce these requirements for each new mapping
	*/

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

		//Failed as couldn't traverse the graph! (i.e. disconnected query)
		if (!found) {
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
	
		if (numOfReqs) {
			std::sort(reqRels, reqRels + numOfReqs, reqRelCompare);

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

//TODO: Remove this!
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