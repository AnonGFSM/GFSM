#pragma once
#include "HnGPU.cuh"
#include <chrono>

//Pretty print the error type
void cudaExit(cudaError_t error) {
    info_printf("\nCuda Error: %s", cudaGetErrorString(error));
    csv_printf("%s,", cudaGetErrorString(error));
    exit(error);
}

namespace HnGPU {

    __device__ uint32_t debugSet = 0;

    /*
    * Used for returning Debug/assert statements without flooding logs
    */
    enum DebugMessage {
        CorruptedData
    };

    __device__ bool debugFetch(DebugMessage msg) {
#ifdef GPUDEBUG
        return !(atomicOr(&debugSet, 1 << msg) & (1 << msg));
#else
        return false;
#endif
    }

    /*
    * Check whether a requirement exists within a row of the CCSR
    */
    __device__ bool reqExists(CCSRSegment* vertexEdge, uint32_t requiredRelation, int32_t requiredVertex, uint32_t len) {
#ifdef SHORTREL

        CCSRSegment* localEdge = vertexEdge;
        for (uint32_t i = 0; i < len; i++, localEdge++) {
            constexpr uint32_t RelMask = CCSRRelation(CCSRRelSize);
            constexpr uint32_t VertMask = CCSRIndex(CCSRVertSize);
            if (((*localEdge) & RelMask) == requiredRelation) {
                if (((*localEdge) & VertMask) == requiredVertex) { // || requiredVertex < 0 Use to be hear??
                    return true;
                }
            }
        }

        return false;
#else
        return 0;
#endif
    }

    /*
    * Convert a number to a valid block count of a given size 
    * (i.e. 242 jobs would require 4 threadblocks of 64 threads to compute)
    */
    __host__ __device__ int32_t getBlockCount(int globalRange, int localRange) {
        return (globalRange / localRange) + 1;
    }

    /*
    * Update the problem description to reflect a new state
    * (Prevents race conditions)
    */
    __device__ void updateHeader(uint32_t* solutionHeader) {
        
        //TODO: This should use volatiles not atomicAdd to ignore cached result (Issue on Turing Codegen)
        solutionHeader[HNFUNC_OFFSET] = atomicAdd(&solutionHeader[HNFUNC_NEWOFFSET], 0);
        solutionHeader[HNFUNC_COUNT] = atomicAdd(&solutionHeader[HNFUNC_NEWCOUNT], 0);
    }

    /*
    * Initialise the FFS 
    */
    __global__ void hnInit(CCSRSegment* codomainEdges, HnFunction* functions, 
                    uint32_t* bBfrFunctionDegrees,
                    uint32_t* solutionHeader,
                    uint32_t* functCounts,
                    uint32_t spacing, uint32_t solutionOffset, uint32_t graphOffset,
                    uint32_t* requirements, uint32_t numOfReqs, uint32_t candidateCount){


        uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;

        if (ix < candidateCount) {

            uint32_t graphIndex = ix * spacing + graphOffset;

            HnFunction* fn = &functions[ix + solutionOffset];
            uint32_t* fnCount = &functCounts[ix + solutionOffset];

            uint32_t* graphRef = &codomainEdges[graphIndex + 1];

            for (int i = 0; i < numOfReqs; i++) {
                if (!reqExists(graphRef, requirements[i*2 + HNREQS_REL], graphIndex, spacing - 1)) {
                    fn->mapping = -1;
                    fn->previous = 0;
                    *fnCount = 0;
                    return;
                }
            }

            fn->mapping = graphIndex;
            fn->previous = 0;
            *fnCount = 1;
            bBfrFunctionDegrees[ix + solutionOffset] = spacing - 1;
        }
    }

    /*
    * Populate the lowest depth of the FFS
    */
    __global__ void hnGen(CCSRSegment* codomainEdges, HnFunction* functions, HnFunction* bBfrFunctions,
                    uint32_t* functionDegrees, uint32_t* functAllocs, 
                    uint32_t* solutionHeader,
                    uint32_t* functCounts,
                    uint32_t threadAlloc, uint32_t firstOccurence,
                    uint32_t batchOffset) {

        uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;

        if (ix < solutionHeader[HNFUNC_COUNT]) {

            uint32_t fnOffset = solutionHeader[HNFUNC_OFFSET];
            uint32_t fnIndex = fnOffset + ix;

            HnFunction fn = functions[fnIndex];

            //Heurstic: Find Closest Connected Vertex to New Vertex in Query
            for (uint32_t i = 1; i < firstOccurence; i++) {
                fn = functions[fn.previous];
            }

            //Generate the new funcs
            uint32_t fnLoc = solutionHeader[HNFUNC_COUNT] + functAllocs[ix];
            uint32_t fnTarget = fnOffset + fnLoc + batchOffset;

            CCSRSegment* vertexEdge = &codomainEdges[fn.mapping];

            HnFunction* fnNew = &bBfrFunctions[fnTarget];

            uint32_t vSize = *vertexEdge;

#ifdef PREDICTCHECK
            if (functionDegrees[ix] != vSize) {
                if (debugFetch(CorruptedData)) {
                    debug_printf("\nCorruption found at %u, %i != %i for (%i, %i)", ix, functionDegrees[ix], vSize, fn.mapping, fn.previous);
                    __trap();
                }
            }
#endif

#ifdef OVERFLOWCHECK
            if (ix) {
                if (functAllocs[ix - 1] > functAllocs[ix]) {
                    if (debugFetch(CorruptedData)) {
                        debug_printf("\nOverflow found at %u, %i > %i", ix, functAllocs[ix - 1] > functAllocs[ix]);
                        __trap();
                    }
                }
            }
#endif

            vertexEdge++;
            uint32_t fns;

            for(fns = 0; fns < vSize; fns++){
                constexpr uint32_t VertMask = CCSRIndex(CCSRVertSize);
                fnNew->mapping = (*vertexEdge) & VertMask;
                fnNew->previous = fnIndex;

                fnNew++;
                vertexEdge++;
            }

            if (ix == solutionHeader[HNFUNC_COUNT] - 1) {
                solutionHeader[HNFUNC_NEWOFFSET] = solutionHeader[HNFUNC_OFFSET] + solutionHeader[HNFUNC_COUNT] + batchOffset;
                solutionHeader[HNFUNC_NEWCOUNT] = functAllocs[ix] + vSize;
            }

            functCounts[ix] = vSize;
        }

    }

    /*
    * Check for bad mappings in the lowest depth of the FFS
    */
    __global__ void hnCheck(CCSRSegment* codomainEdges, HnFunction* functions, HnFunction* bBfrFunctions,
                    uint32_t* bBfrFunctionDegrees,
                    uint32_t* solutionHeader,
                    uint32_t* functCounts, uint32_t* functOffsets, 
                    uint32_t* requirements, uint32_t numOfReqs, uint32_t reqOffset, uint32_t depth,
                    uint32_t nextFirstOccurence) {

        uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;

        if (ix < solutionHeader[HNFUNC_COUNT]) {

            uint32_t fnIndex = ix + solutionHeader[HNFUNC_OFFSET];

            HnFunction* fn = &bBfrFunctions[fnIndex];

            HnFunction* cFn = fn;

            int32_t newVertex = fn->mapping;

#ifdef GPUDEBUG
            if (fn->mapping == -1) {
                debug_printf("\nBad Thread Found! %u", ix);
                __trap();
            }
#endif

            if (nextFirstOccurence == 1) {
                bBfrFunctionDegrees[ix] = codomainEdges[fn->mapping];
            }


            //Observes the size for the lookahead mechanism (for the next generation)
            //and checks whether the new mapping preserves injectiveness
            for (int32_t i = 0; i <= depth; i++) {

                fn = &functions[fn->previous];
#ifdef ISOSOLVE
                if (newVertex == fn->mapping) {
                    cFn->mapping = -1;
                    functCounts[ix] = 0;
                    return;
                }
#endif
                if (i == nextFirstOccurence - 2) {
                    bBfrFunctionDegrees[ix] = codomainEdges[fn->mapping];
                }
            }


            //Test whether the requirements are observed in the mapping's CCSR row
            fn = cFn;
            uint32_t foundLocs = numOfReqs;

            for (uint32_t i = 0; i < numOfReqs * 2; i += 2) {
                uint32_t reqDist = requirements[i + HNREQS_DIST + reqOffset];
                uint32_t reqRel = requirements[i + HNREQS_REL + reqOffset];

                for (uint32_t i2 = 0; i2 < reqDist; i2++) {
                    fn = &functions[fn->previous];
                }

                foundLocs -= reqExists(&codomainEdges[fn->mapping + 1], reqRel, newVertex, codomainEdges[fn->mapping]);
            }

            if (foundLocs) {
                cFn->mapping = -1;
                functCounts[ix] = 0;
            }
            else {
                functCounts[ix] = 1;
            }
        }
    }

    /*
    * Special function used for batching
    * 
    * Subtracts a value from all values in a buffer
    */
    __global__ void hnRem(uint32_t* bfr, uint32_t val, uint32_t count) {
        uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
        if (ix < count) {
            bfr[ix] = bfr[ix] - val;
        }
    }

    /*
    * Special function used for batching
    * 
    * Recomputes the Degrees (for the lookahead mechanism) of partial matches
    */
    __global__ void hnDeg(HnFunction* functions, CCSRSegment* codomainEdges, 
        uint32_t* functionDegrees, uint32_t* solutionHeader,
        uint32_t depth, 
        uint32_t nextFirstOccurence) {

        uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;

        if (ix < solutionHeader[HNFUNC_COUNT]) {
            HnFunction* fn = &functions[ix + solutionHeader[HNFUNC_OFFSET]];
            if (nextFirstOccurence == 1) {
                functionDegrees[ix] = codomainEdges[fn->mapping];
            }
            else {
                for (uint32_t i = 0; i <= depth; i++) {
                    fn = &functions[fn->previous];

                    if (i == nextFirstOccurence - 2) {
                        functionDegrees[ix] = codomainEdges[fn->mapping];
                    }
                }
            }
        }
    }

    /*
    * Removes all flagged mappings from the FFS 
    * (by moving the ones which are valid, to be written serially)
    */
    __global__ void hnRemove(HnFunction* functions, HnFunction* bBfrFunctions,
                    uint32_t* functionDegrees, uint32_t* bBfrFunctionDegrees,
                    uint32_t* solutionHeader,
                    uint32_t* functOffsets,
                    uint32_t batchOffset) {

        uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;

        if (ix < solutionHeader[HNFUNC_COUNT]) {
            uint32_t fnOffset = solutionHeader[HNFUNC_OFFSET];
            uint32_t fnIndex = fnOffset + ix;
            uint32_t localOffset = functOffsets[ix];
            uint32_t newFnIndex = fnOffset + localOffset + batchOffset;
            HnFunction fn = bBfrFunctions[fnIndex];

            if (fn.mapping != -1) {
                functions[newFnIndex] = fn;
                functionDegrees[localOffset] = bBfrFunctionDegrees[ix];
            }

            if (ix == solutionHeader[HNFUNC_COUNT] - 1) {
                solutionHeader[HNFUNC_NEWCOUNT] = functOffsets[ix] + (fn.mapping != -1);
            }
        }

    }

    /*
    * Calls CUB implementation of Device-Wide Exclusive Scan
    */
    __device__ cudaError_t hnScan(uint32_t* scanBfr, uint32_t* scanBfrTemp, size_t tempBfrSize, 
                uint32_t* functCounts, uint32_t* solutionHeader, size_t SCAN_LIMIT) {

        if (solutionHeader[HNFUNC_COUNT] > SCAN_LIMIT) {
            debug_printf("\nScan Overflow: %i", solutionHeader[HNFUNC_COUNT]);
            solutionHeader[HNFUNC_COUNT] = DEBUGCOUNT;
            return cudaErrorInvalidValue;
        }
        cub::DeviceScan::ExclusiveSum(scanBfrTemp, tempBfrSize, functCounts, scanBfr, solutionHeader[HNFUNC_COUNT]);
        return cudaDeviceSynchronize();
    }

    /*
    * Unused
    * Calls CUB implementation of Device-Wide Radix Sort
    */
    template <class K, class V>
    __device__ cudaError_t hnSort(uint32_t* sortBfrTemp, size_t tempBfrSize, K* keys, K* keysSort, V* values, V* valuesSort,
        uint32_t* solutionHeader, size_t SCAN_LIMIT) {
        if (solutionHeader[HNFUNC_COUNT] > SCAN_LIMIT) {
            //printf("\nSort Overflow: %i", solutionHeader[HNFUNC_COUNT]);
            solutionHeader[HNFUNC_COUNT] = DEBUGCOUNT;
            return cudaErrorInvalidValue;
        }

        cub::DeviceRadixSort::SortPairs(sortBfrTemp, tempBfrSize, keys, keysSort, values, valuesSort, solutionHeader[HNFUNC_COUNT]);
        return cudaDeviceSynchronize();
    }

    /*
    * The main loop of the task Kernel
    */
    __device__ __forceinline__ bool hnLoop(CCSRSegment* codomainEdges, HnFunction* functions, HnFunction* bBfrFunctions,
        uint32_t* functionDegrees, uint32_t* bBfrFunctionDegrees,
        uint32_t* solutionHeader,
        uint32_t* staggers, uint32_t staggerLen,
        uint32_t* functCounts, uint32_t* scanBfr, uint32_t* scanBfrTemp,
        uint32_t* requirements, uint32_t* requirementHeader,
        uint32_t batchOffset,
        uint32_t depthTarget, uint32_t maxValency,
        uint32_t depth, 
        size_t SOLN_LIMIT, size_t SCAN_LIMIT) {

        uint32_t reqIndex = ((depth + 1) * 3);
        uint32_t nextReqIndex = reqIndex + (depth != depthTarget) * 3;

        //HnGen - Generate candidates

        uint32_t blockCount = getBlockCount(solutionHeader[HNFUNC_COUNT], HBLOCKSIZE);

        hnGen << < blockCount, HBLOCKSIZE >> > (codomainEdges, functions, bBfrFunctions, functionDegrees, scanBfr, solutionHeader, functCounts,
            maxValency, requirementHeader[HNREQS_H_FIRSTQVERTEX + reqIndex], batchOffset);

        if (cudaSuccess != cudaDeviceSynchronize()) return true;

        updateHeader(solutionHeader);

        debug_printf("\nHnGen -> %u", solutionHeader[HNFUNC_COUNT]);

        blockCount = getBlockCount(solutionHeader[HNFUNC_COUNT], HBLOCKSIZE);

        //HnCheck - Check candidates

        hnCheck << < blockCount, HBLOCKSIZE >> > (codomainEdges, functions, bBfrFunctions, bBfrFunctionDegrees, solutionHeader, functCounts, scanBfr,
            requirements, requirementHeader[HNREQS_H_NUM + reqIndex], requirementHeader[HNREQS_H_INDEX + reqIndex], depth,
            requirementHeader[HNREQS_H_FIRSTQVERTEX + nextReqIndex]);

        if (cudaSuccess != cudaDeviceSynchronize()) return true;


        //HnRemove - Remove flagged candidates

        debug_printf("\nHnCheck -> %u", solutionHeader[HNFUNC_COUNT]);

        if (cudaSuccess != hnScan(scanBfr, scanBfrTemp, SCAN_LIMIT * sizeof(int), functCounts, solutionHeader, SCAN_LIMIT)) return true;

        blockCount = getBlockCount(solutionHeader[HNFUNC_COUNT], HBLOCKSIZE);

        if (batchOffset) {
            uint32_t funcCount = solutionHeader[HNFUNC_COUNT];
            if (scanBfr[funcCount - 1] + functCounts[funcCount - 1] + solutionHeader[HNFUNC_OFFSET] + batchOffset >= SOLN_LIMIT) {
                debug_printf("\nBatch Overflow scan: %u, count: %u, offset: %u, batchOffset: %u", scanBfr[funcCount - 1], functCounts[funcCount - 1], solutionHeader[HNFUNC_OFFSET], batchOffset);
                return true;
            }
        }

        hnRemove << < blockCount, HBLOCKSIZE >> > (functions, bBfrFunctions, functionDegrees, bBfrFunctionDegrees, solutionHeader, scanBfr, batchOffset);

        if (cudaSuccess != cudaDeviceSynchronize()) return true;

        updateHeader(solutionHeader);

        debug_printf("\nHnRemove -> %u", solutionHeader[HNFUNC_COUNT]);
        pcsv_printf("HnStep: %u (%u),", solutionHeader[HNFUNC_COUNT], depth);

        //TODO: Implement at some point, it doesn't work! Should help accelerate certain workloads
        //if (cudaSuccess != hnSort(scanBfr, scanBfrTemp, MAXSCANSIZE * sizeof(int), functionDegrees, scanBfr, solutionHeader)) return true;

        if (cudaSuccess != hnScan(scanBfr, scanBfrTemp, SCAN_LIMIT * sizeof(int), functionDegrees, solutionHeader, SCAN_LIMIT)) return true;

        return false;
    }

    __device__ uint32_t divCeil(uint32_t l, uint32_t r) {
        return (l / r) + ((l / r) != 0);
    }

#define BATCHINGSPACE 1024 * 16

    /*
    * The Job launched for the task kernel --
    * 
    * - Initialises solution
    * - Then chooses between:
    *       - Computing main loop 
    *       - Using batching (for very large problems)
    */
    __global__ void hnSolve(CCSRSegment* codomainEdges, HnFunction* functions, HnFunction* bBfrFunctions,
                    uint32_t* functionDegrees, uint32_t* bBfrFunctionDegrees,
                    uint32_t* solutionHeader,
                    uint32_t* staggers, uint32_t staggerLen,
                    uint32_t* functCounts, uint32_t* scanBfr, uint32_t* scanBfrBack, uint32_t* scanBfrTemp,
                    uint32_t* requirements, uint32_t* requirementHeader,
                    uint32_t depthTarget, uint32_t maxValency, 
                    size_t SOLN_LIMIT, size_t SCAN_LIMIT){
        int blockCount;

        uint32_t graphOffset = 0;
        uint32_t solutionOffset = 0;
        uint32_t reqIndex = 0;
        uint32_t stagger, genFuncs;

        solutionHeader[HNFUNC_OFFSET] = 0;

        //Initialise solution
        //As CCSR rows decrease across the CCSR, launch a workload for each length of row 
        //(highest -> lowest)
        for (int spacing = staggerLen+1; spacing > 1; spacing--) {

            stagger = staggers[spacing -2];

            genFuncs = stagger / spacing;

            if (genFuncs) {
                blockCount = getBlockCount(genFuncs, HBLOCKSIZE);
                hnInit <<< blockCount, HBLOCKSIZE >>>(codomainEdges, bBfrFunctions, bBfrFunctionDegrees, solutionHeader, functCounts,
                    spacing, solutionOffset, graphOffset,
                    requirements, requirementHeader[HNREQS_H_NUM], genFuncs);
            }

            solutionOffset += genFuncs;

            if (stagger) {
                graphOffset += stagger;
            }
            
            solutionHeader[HNFUNC_COUNT] += genFuncs;
        }

        if (cudaSuccess != cudaDeviceSynchronize()) __trap();

        debug_printf("\nHnInit -> %u", solutionHeader[HNFUNC_COUNT]);

        if (cudaSuccess != hnScan(scanBfr, scanBfrTemp, SCAN_LIMIT * sizeof(int), functCounts, solutionHeader, SCAN_LIMIT)) return;

        blockCount = getBlockCount(solutionHeader[HNFUNC_COUNT], HBLOCKSIZE);

        hnRemove <<< blockCount, HBLOCKSIZE >>> (functions, bBfrFunctions, functionDegrees, bBfrFunctionDegrees, solutionHeader, scanBfr, 0);

        if (cudaSuccess != cudaDeviceSynchronize()) __trap();

        updateHeader(solutionHeader);

        debug_printf("\nHnRemove -> %u", solutionHeader[HNFUNC_COUNT]);

        if (cudaSuccess != hnScan(scanBfr, scanBfrTemp, SCAN_LIMIT * sizeof(int), functionDegrees, solutionHeader, SCAN_LIMIT)) return;

        solutionHeader[HNFUNC_COUNT] = solutionHeader[HNFUNC_NEWCOUNT];

        //Repeat the following for each remaining requirement
        for (int depth = 0; depth < depthTarget; depth++) {

            uint32_t funcCount = solutionHeader[HNFUNC_COUNT];
            uint32_t predFuncCount = scanBfr[funcCount - 1] + functionDegrees[funcCount-1];
            uint32_t funcOffset = solutionHeader[HNFUNC_OFFSET];
            uint32_t predCeil = predFuncCount + funcCount + funcOffset;
            debug_printf("\nHnPred -> %u", predFuncCount);
            pcsv_printf("HnPred: %u (%u),", predFuncCount, depth);

            //Check to see whether we are going to overflow
            //If we will, lets try batching to solve this
            if (predCeil >= SOLN_LIMIT) {
                debug_printf("\nOVERFLOW DETECTED\nDepth: %i, Offset: %u, Count: %u Target: %u", depth, funcOffset, funcCount, predFuncCount);
                debug_printf("\nFree: %u, Max: %u, Required: %u", SOLN_LIMIT - (funcCount + funcOffset), SOLN_LIMIT, predCeil);
#ifdef BATCHING
                uint32_t space = SOLN_LIMIT - funcOffset - funcCount - 1;
                uint32_t batchSize = funcCount / (divCeil(predFuncCount, space));
                debug_printf("\nBatch Size %u (%u / %u)", batchSize, funcCount, divCeil(predFuncCount, space));

                if (batchSize > BATCHINGSPACE) {
                    uint32_t newFuncCount = 0;
                    uint32_t filteredCandidates = 0;
                    uint32_t pSize;

                    for (uint32_t pF = 0; pF < funcCount; pF += pSize) {

                        //Look for a small enough (but large) batch we could fit
                        pSize = batchSize;
                        if (pSize > funcCount - pF) {
                            pSize = funcCount - pF;
                        }

                        const uint32_t safety = 4;

                        for (uint32_t i = 0; i < safety; i++) {
                            if (scanBfr[pF + pSize - 1] == 0) {
                                return;
                            }

                            if (scanBfr[pF+ pSize-1] - scanBfr[pF] > space - newFuncCount) {
                                pSize /= 2;
                            }
                            else {
                                break;
                            }

                            if (i == safety - 1 || pSize == 0) { //This should be moved above
                                return;
                            }
                        }

                        uint32_t expectedDif = scanBfr[pF + pSize - 1] - scanBfr[pF];

                        solutionHeader[HNFUNC_COUNT] = pSize;

                        cudaMemcpyAsync(scanBfrBack, scanBfr + pF, pSize * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

                        //As only batches of the problem can be computed at once we recalculate
                        //the current functions degree each time (as we cannot store all of them)
                        if (filteredCandidates) {
                            blockCount = getBlockCount(solutionHeader[HNFUNC_COUNT], HBLOCKSIZE);
                            hnRem << <blockCount, HBLOCKSIZE >> > (scanBfrBack, filteredCandidates, solutionHeader[HNFUNC_COUNT]);
                            hnDeg << <blockCount, HBLOCKSIZE >> > (functions, codomainEdges, functionDegrees, solutionHeader, depth-1, requirementHeader[HNREQS_H_FIRSTQVERTEX + (depth+1)*3]);
                        }

                        if(pF + pSize < funcCount)
                            filteredCandidates = scanBfr[pF + pSize];

                        if (cudaDeviceSynchronize())
                            return; //This will likely never return.. Grid will exit (Crash) before it hits this

                        if (scanBfrBack[pSize - 1] != expectedDif) {
                            __trap();
                        }

                        uint32_t prevOffset = solutionHeader[HNFUNC_OFFSET];

                        if (hnLoop(codomainEdges, functions, bBfrFunctions,
                            functionDegrees, bBfrFunctionDegrees,
                            solutionHeader,
                            staggers, staggerLen,
                            functCounts, scanBfrBack, scanBfrTemp,
                            requirements, requirementHeader,
                            newFuncCount + funcCount - pF - pSize,
                            depthTarget, maxValency,
                            depth,
                            SOLN_LIMIT, SCAN_LIMIT)) {
                            solutionHeader[HNFUNC_COUNT] = DEBUGCOUNT;
                            debug_printf("\nGeneral loop failure %i", depth);
                            csv_printf("Loop failure %i,", depth);
                            return;
                        }

                        newFuncCount += solutionHeader[HNFUNC_COUNT];

                        pcsv_printf("HnBatch: %u (+ %u),", newFuncCount, solutionHeader[HNFUNC_COUNT]);

                        solutionHeader[HNFUNC_OFFSET] = prevOffset + pSize;
                    }

                    solutionHeader[HNFUNC_COUNT] = newFuncCount;
                    uint32_t newFuncOffset = solutionHeader[HNFUNC_OFFSET];
                    solutionHeader[HNFUNC_OFFSET] = funcOffset;


                    hnDeg << <getBlockCount(newFuncCount, HBLOCKSIZE), HBLOCKSIZE >> > (functions, codomainEdges, functionDegrees, solutionHeader, depth, requirementHeader[HNREQS_H_FIRSTQVERTEX + (depth + 1 + depth != depthTarget) * 3]);
                    if (cudaSuccess != hnScan(scanBfr, scanBfrTemp, MAXSCANSIZE * sizeof(int), functionDegrees, solutionHeader, SCAN_LIMIT)) return;

                    solutionHeader[HNFUNC_OFFSET] = newFuncOffset;
                }
                else {
                    return;
                }
#else
                if (SOLN_LIMIT == MAXSOLNSIZE) {
                    solutionHeader[HNFUNC_COUNT] = DEBUGCOUNT;
                }
                else {
                    csv_printf("Failure: Heuristic generated %u functions,", predFuncCount);
                    __trap();
                }
                return;
#endif
            } else {

                //Most Common case, just enter the loop and launch the work
                if (hnLoop(codomainEdges, functions, bBfrFunctions,
                    functionDegrees, bBfrFunctionDegrees,
                    solutionHeader,
                    staggers, staggerLen,
                    functCounts, scanBfr, scanBfrTemp,
                    requirements, requirementHeader,
                    0,
                    depthTarget, maxValency,
                    depth,
                    SOLN_LIMIT, SCAN_LIMIT)) {
                        solutionHeader[HNFUNC_COUNT] = DEBUGCOUNT;

                        //Generally these debug messages are never seen, as errors cause a early grid exit
                        //TODO: Possibly remove them?
                        debug_printf("General loop failure %i", depth);
                        csv_printf("Loop failure %i,", depth);
                        return;
                }
            }

            if (volatile int count = solutionHeader[HNFUNC_COUNT] == 0) {
                return;
            }
        }

    }

    /*
    * Converts the FFS into a table form (respecting the original input data)
    */
    __global__ void hnWrite(CCSRSegment* codomainEdges, HnFunction* functions, uint32_t* functionInfo,
                        uint32_t* inverseDataVerticeLocs, int32_t* queryMappingData,
                        CCSR::StaggerInfo* staggersInfoGPU,
                        uint32_t functionCount, uint32_t functionOffset,
                        uint32_t querySize) {
        uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t* fInfo = &functionInfo[ix * querySize];

        if (ix < functionCount) {
            HnFunction function = functions[ix + functionOffset];
            for (uint32_t i = 0; i < querySize; i++) {

                //Convert mapping to be in terms of the original input data (instead of CCSR location)
                uint32_t mapping = function.mapping;
                uint32_t degree = codomainEdges[mapping];
                CCSR::StaggerInfo staggerInfoGPU = staggersInfoGPU[degree - 1];
                uint32_t normalisedIndex = staggerInfoGPU.normOffset + (mapping - staggerInfoGPU.offset) / (degree + 1);
                fInfo[(uint32_t)queryMappingData[i]] = inverseDataVerticeLocs[normalisedIndex];

                //Step up the FFS
                function = functions[function.previous];
            }

        }

    }
}

namespace HnSetup {


    /*
    * Solves a subgraph matching problem between two CCSR graphs
    * 
    * Requires HnSetup::preinit called before (or cuda buffers won't be allocated)
    */
    __host__ std::vector<HnSolution>* solve(const CCSR::CCSRGraph& ccsrQuery, const CCSR::CCSRGraph& ccsrData,
        const CCSR::CCSRStagger& queryStagger, const CCSR::CCSRStagger& dataStagger) {

        auto start = std::chrono::steady_clock::now();

        uint32_t* requirements, * requirementHeader;

        int32_t* mappingData;

        uint32_t maxValencyQuery = preProcessQuery(ccsrQuery, &requirements, &requirementHeader, &mappingData);

        uint32_t maxValencyData = CCSR::getMaxValency(ccsrData);


        HnFunction* solutions;
        uint32_t* solutionHeader;

        gpuLaunch(ccsrQuery, ccsrData, dataStagger,
            requirements, requirementHeader,
            maxValencyQuery, maxValencyData, mappingData,
            &solutions, &solutionHeader,
            MEMLIMIT, MAXSOLNSIZE, MAXSCANSIZE);

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        info_printf("\nGPU Time: %fs\n", elapsed_seconds.count());
        csv_printf("%fs,", elapsed_seconds.count());

        return NULL;
    }

    /*
    * Preprocess
    * 
    * - Sets up requirements for the query graph -- 
    *   - Chooses a matching order
    *   - Sets up the requirements buffer data
    *   - Max degree in the query graph
    */
    __host__ uint32_t preProcessQuery(CCSR::CCSRGraph query, uint32_t** requirements, uint32_t** requirementHeader, int32_t** mappingData) {

        uint32_t maxValency = CCSR::getMaxValency(query);

        *requirements = (uint32_t*)calloc(2 * maxValency * query.count, sizeof(uint32_t));
        *requirementHeader = (uint32_t*)calloc(3 * query.count, sizeof(uint32_t));

        *mappingData = (int32_t*) malloc(sizeof(int32_t) * query.count);

        memset((*mappingData), -1, sizeof(int32_t) * query.count);
        uint32_t degVal = query.segments[0];
        uint32_t loc = 0;

        for (uint32_t i = 0; i < query.size; i+=query.segments[i]+1) {
            if (degVal < query.segments[i]) {
                degVal = query.segments[i];
                loc = i;
            }
        }
        
        (*mappingData)[loc] = 0;

        for (int depth = 0; depth < query.count; depth++) {
            CCSR::genereateCCSRRequirements(query, *requirements, *requirementHeader, maxValency, depth, *mappingData);
        }


        int32_t* tempMappingData = new int32_t[query.count]{};
        uint32_t* orderingData = new uint32_t[query.count]{};

        for (uint32_t i = 0; i < query.count; i++) {
            for (uint32_t i2 = 0; i2 < query.count; i2++) {
                if ((*mappingData)[i2] == i + 1) {
                    orderingData[i] = i2;
                }
            }
        }

        memcpy(*mappingData, tempMappingData, sizeof(uint32_t) * query.count);

        delete [] tempMappingData;
        delete [] orderingData;

#ifdef QUERYDATAPRINT
        info_printf("\nMaxValency %i, Query Height %i: ", maxValency, query.count);

        for (uint32_t i = 0; i < maxValency * query.count; i++) {
            info_printf("\nReqDist %i, ", (*requirements)[i * 2]);
            info_printf("ReqValue %i (%i), ", (*requirements)[i * 2 + 1] >> CCSRVertSize, (*requirements)[i * 2 + 1]);
        }
        info_printf("\n\n", 0);
        for (uint32_t i = 0; i < query.count; i++) {
            info_printf("\nNumReqs %i, ", (*requirementHeader)[3 * i]);
            info_printf("ReqIndex %i, ", (*requirementHeader)[3 * i + 1]);
            info_printf("FirstQueryDepth %i, ", (*requirementHeader)[3 * i + 2]);
        }
#endif

        for (uint32_t i = 0; i < maxValency * query.count; i++) {
            if ((*requirements)[i * 2] > query.count) {
                info_printf("\nPrecompiler Failure: Query Gen,", 0);
                pcsv_printf("Precompiler Failure: Query Gen,", 0);
                exit(1);
            }
        }

        return maxValency;
    }

    /*
    * General Allocation Helper Function
    * 
    * Uses CudaMalloc not AsyncCudaMalloc currently as benchmarks were done on Cuda 11.2 
    * (Could be changed to cudaMallocAsync for faster results).
    */

    template<class T>
    static __inline__ __host__ cudaError_t s_cudaMallocAsync(T** ptr, size_t size, cudaStream_t stream, size_t* total) {
        if (total) {
            (*total) += size;
        }

        return cudaMalloc(ptr, size);
    }

    size_t totalAlloc = 0;
    HnFunction* frontBfr, * backBfr;
    uint32_t* functCountsBfr, * scanBfr, * scanBfrBack, * scanBfrTemp,  * functionDegrees, * bBfrFunctionDegrees;

    //Exists to flush queue and force early malloc (on Ampere Devices). Not required on Turing
    __global__ void dummy() {

    }

    /*
    * Pre allocates all main GPU buffers (FFS storage and Exclusive scan results)
    */

    __host__ void preinit(size_t MEM_LIMIT, size_t SOLN_LIMIT, size_t SCAN_LIMIT) {
        auto start = std::chrono::steady_clock::now();
        s_cudaMallocAsync(&frontBfr, sizeof(HnFunction) * SOLN_LIMIT, 0, &totalAlloc);
        s_cudaMallocAsync(&backBfr, sizeof(HnFunction) * SOLN_LIMIT, 0, &totalAlloc);
        s_cudaMallocAsync(&functCountsBfr, sizeof(uint32_t) * SCAN_LIMIT, 0, &totalAlloc);
        s_cudaMallocAsync(&functionDegrees, sizeof(uint32_t) * SCAN_LIMIT, 0, &totalAlloc);
        s_cudaMallocAsync(&bBfrFunctionDegrees, sizeof(uint32_t) * SCAN_LIMIT, 0, &totalAlloc);
        s_cudaMallocAsync(&scanBfr, sizeof(uint32_t) * SCAN_LIMIT, 0, &totalAlloc);
        s_cudaMallocAsync(&scanBfrBack, sizeof(uint32_t) * SCAN_LIMIT, 0, &totalAlloc);
        s_cudaMallocAsync(&scanBfrTemp, sizeof(uint32_t) * SCAN_LIMIT, 0, &totalAlloc);
        dummy << <1, 1 >> > ();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        info_printf("\nPreinit Time: %fs\n", elapsed_seconds.count());
        csv_printf("%fs,", elapsed_seconds.count());
    }

    /*
    * Launches all GPU work and allocates memory for CCSR data (device-side)
    */
    __host__ void gpuLaunch(const CCSR::CCSRGraph& query, const CCSR::CCSRGraph& data, const CCSR::CCSRStagger& dataStagger,
        uint32_t* requirements, uint32_t* requirementHeader,
        uint32_t maxValencyQuery, uint32_t maxValencyData, int32_t* mappingData,
        HnFunction** solutionCPU, uint32_t** solutionHeaderCPU,
        size_t MEM_LIMIT, size_t SOLN_LIMIT, size_t SCAN_LIMIT) {

        CCSRSegment *queryGPU, *dataGPU;
        uint32_t *requirementsGPU, *requirementHeaderGPU, *solutionHeader;
        uint32_t *staggersGPU;

        auto start = std::chrono::steady_clock::now();
        s_cudaMallocAsync(&queryGPU, sizeof(CCSRSegment) * query.size, 0, &totalAlloc);
        s_cudaMallocAsync(&dataGPU, sizeof(CCSRSegment) * data.size, 0, &totalAlloc);
        s_cudaMallocAsync(&solutionHeader, sizeof(uint32_t) * 6, 0, &totalAlloc);

        s_cudaMallocAsync(&requirementsGPU, sizeof(uint32_t) * 2 * maxValencyQuery * query.count, 0, &totalAlloc);
        s_cudaMallocAsync(&requirementHeaderGPU, sizeof(uint32_t) * 3 * query.count, 0, &totalAlloc);

        s_cudaMallocAsync(&staggersGPU, sizeof(uint32_t) * dataStagger.size, 0, &totalAlloc);

        cudaError_t cudaStatus = cudaDeviceSynchronize();
        if (cudaSuccess != cudaStatus) cudaExit(cudaStatus);

        cudaMemcpyAsync(queryGPU, query.segments, sizeof(CCSRSegment) * query.size, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(dataGPU, data.segments, sizeof(CCSRSegment) * data.size, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(requirementsGPU, requirements, sizeof(uint32_t) * 2 * maxValencyQuery * query.count, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(requirementHeaderGPU, requirementHeader, sizeof(uint32_t) * 3 * query.count, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(staggersGPU, dataStagger.staggers, sizeof(uint32_t) * dataStagger.size, cudaMemcpyHostToDevice);

        uint32_t* invDataVerticeLocsGPU;
        int32_t* queryMappingDataGPU;
        CCSR::StaggerInfo* staggersInfoGPU;

        s_cudaMallocAsync(&queryMappingDataGPU, sizeof(int32_t) * query.count, 0, NULL);
        s_cudaMallocAsync(&invDataVerticeLocsGPU, sizeof(uint32_t) * data.count, 0, NULL);
        s_cudaMallocAsync(&staggersInfoGPU, sizeof(CCSR::StaggerInfo) * maxValencyData, 0, NULL);

        cudaMemcpyAsync(queryMappingDataGPU, mappingData, sizeof(int32_t) * query.count, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(invDataVerticeLocsGPU, data.invVerticeLocs, sizeof(uint32_t) * data.count, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(staggersInfoGPU, dataStagger.staggersInfo, sizeof(CCSR::StaggerInfo) * maxValencyData, cudaMemcpyHostToDevice);

        cudaStatus = cudaDeviceSynchronize();

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds2 = end - start;
        info_printf("\nSetup Time: %fs\n", elapsed_seconds2.count());

        start = std::chrono::steady_clock::now();

        if (totalAlloc > MEM_LIMIT) {
            info_printf("\nToo Much Memory Requested: %zu", totalAlloc);
            exit(1);
        }
        else {
            info_printf("\nUsing Memory: %zu bytes", totalAlloc);
        }

        if (cudaSuccess != cudaStatus) cudaExit(cudaStatus);

        *solutionCPU = (HnFunction*)malloc(sizeof(HnFunction) * SOLN_LIMIT);
        *solutionHeaderCPU = (uint32_t*)malloc(sizeof(uint32_t) * 6);

        info_printf("\nSolve Kernel Start:", 0);

        HnGPU::hnSolve <<< 1, 1 >>> 
            (dataGPU, frontBfr, backBfr, 
            functionDegrees, bBfrFunctionDegrees,
            solutionHeader,
            staggersGPU, dataStagger.size,
            functCountsBfr, scanBfr, scanBfrBack, scanBfrTemp,
            requirementsGPU, requirementHeaderGPU, 
            query.count - 1, maxValencyData,
            SOLN_LIMIT, SCAN_LIMIT);

        cudaStatus = cudaDeviceSynchronize();

        if (cudaSuccess != cudaStatus) {
            cudaExit(cudaStatus);
        }

        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds3 = end - start;

        info_printf("\nKernel Time: %fs\n", elapsed_seconds3.count());
        csv_printf("%fs,", elapsed_seconds3.count());

        start = std::chrono::steady_clock::now();

        cudaMemcpy(*solutionHeaderCPU, solutionHeader, sizeof(uint32_t) * 6, cudaMemcpyDeviceToHost);

        if ((*solutionHeaderCPU)[HNFUNC_COUNT] == DEBUGCOUNT) { // Memory Limit Failure
            if (MEM_LIMIT == MEMLIMIT) {
                csv_printf("%s", "Reallocating,");
                info_printf("%s", "\nReallocating");

                cudaFree(queryGPU);
                cudaFree(backBfr);
                cudaFree(functCountsBfr);
                cudaFree(scanBfr);
                cudaFree(scanBfrBack);
                cudaFree(scanBfrTemp);
                cudaFree(requirementsGPU);
                cudaFree(requirementHeaderGPU);
                cudaFree(staggersGPU);
                cudaFree(frontBfr);
                cudaFree(dataGPU);
                cudaFree(solutionHeader);
                cudaFree(queryMappingDataGPU);
                cudaFree(invDataVerticeLocsGPU);
                cudaFree(staggersInfoGPU);

                (totalAlloc) = 0;

                preinit(UPPERMEMLIMIT, SOLN_LIMIT * 2, SCAN_LIMIT * 2);
                gpuLaunch(query, data, dataStagger,
                    requirements, requirementHeader,
                    maxValencyQuery, maxValencyData, mappingData,
                    solutionCPU, solutionHeaderCPU,
                    UPPERMEMLIMIT, SOLN_LIMIT * 2, SCAN_LIMIT * 2); 

                end = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed_secondsR = end - start;
                info_printf("\nReallocated GPU Time: %fs\n", elapsed_secondsR.count());
            }
            return;
        }
        else {
            if (MEM_LIMIT == MEMLIMIT) {
                csv_printf("%s", "n/a,n/a,");
            }
        }
        
        info_printf("\nFunction Count: %i", (*solutionHeaderCPU)[HNFUNC_COUNT]);
        csv_printf("%i,", (*solutionHeaderCPU)[HNFUNC_COUNT]);

        cudaFree(queryGPU);
        cudaFree(backBfr);
        cudaFree(functCountsBfr);
        cudaFree(scanBfr);
        cudaFree(scanBfrBack);
        cudaFree(scanBfrTemp);
        cudaFree(requirementsGPU);
        cudaFree(requirementHeaderGPU);
        cudaFree(staggersGPU);
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds4 = end - start;
        info_printf("\nResource End Time: %fs\n", elapsed_seconds4.count());

        uint32_t* functionInfoGPU;

        start = std::chrono::steady_clock::now();

        s_cudaMallocAsync(&functionInfoGPU, sizeof(uint32_t) * (*solutionHeaderCPU)[HNFUNC_COUNT] * query.count, 0, NULL);

    
        int32_t blockCount = HnGPU::getBlockCount((*solutionHeaderCPU)[HNFUNC_COUNT], HBLOCKSIZE);

        cudaStatus = cudaDeviceSynchronize();
        if (cudaSuccess != cudaStatus) cudaExit(cudaStatus);

        start = std::chrono::steady_clock::now();

        HnGPU::hnWrite << <blockCount, HBLOCKSIZE >> > (dataGPU, frontBfr, functionInfoGPU,
            invDataVerticeLocsGPU, queryMappingDataGPU,
            staggersInfoGPU,
            (*solutionHeaderCPU)[HNFUNC_COUNT], (*solutionHeaderCPU)[HNFUNC_OFFSET],
            query.count);

        cudaStatus = cudaDeviceSynchronize();
        if (cudaSuccess != cudaStatus) cudaExit(cudaStatus);

        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds5 = end - start;

        info_printf("\nSolution Creation End Time: %fs\n", elapsed_seconds5.count());
        csv_printf("%fs,", elapsed_seconds5.count());

        info_printf("\nhnWrite Completed", 0);

        uint32_t* functionInfo = (uint32_t*)malloc((*solutionHeaderCPU)[HNFUNC_COUNT] * query.count * sizeof(uint32_t));

        cudaMemcpyAsync(functionInfo, functionInfoGPU, sizeof(uint32_t) * (*solutionHeaderCPU)[HNFUNC_COUNT] * query.count, cudaMemcpyDeviceToHost);
        cudaFreeAsync(frontBfr, 0);
        cudaFreeAsync(dataGPU, 0);
        cudaFreeAsync(solutionHeader, 0);
        cudaFreeAsync(functionInfoGPU, 0);
        cudaFreeAsync(queryMappingDataGPU, 0);
        cudaFreeAsync(invDataVerticeLocsGPU, 0);
        cudaFreeAsync(staggersInfoGPU, 0);

        cudaStatus = cudaDeviceSynchronize();
        if (cudaSuccess != cudaStatus) cudaExit(cudaStatus);
        
        //Commented out, uncomment this if you need output to file
        //functionsToFile(functionInfo, query.count, (*solutionHeaderCPU)[HNFUNC_COUNT]);
        //functionsToFilePretty(functionInfo, query.count, (*solutionHeaderCPU)[HNFUNC_COUNT]);
    }


    /*
    * Pretty Print the Final Result
    */
    __host__ void printFunctions(uint32_t* functionInfo, uint32_t width, uint32_t count) {
        printf("\nFunctions");

        for (uint32_t i = 0; i < count * width; ) {
            printf("\nFunction %lu: ", i / width);
            for (uint32_t i2 = 0; i2 < width; i2++, i++) {
                printf("(%lu, %lu), ", i2, functionInfo[i]);
            }
        }
    }
    
    //Return a file output (as binary)
    __host__ uint32_t functionsToFile(uint32_t* functionInfo, uint32_t width, uint32_t count) {
        std::ofstream os("output.fs", std::ofstream::out | std::ofstream::trunc | std::ios::binary);

        os.write((char*) &count, sizeof(uint32_t));
        os.write((char*) &width, sizeof(uint32_t));
        os.write((char*) functionInfo, count * width * sizeof(uint32_t));

        os.close();

        return false;
    }

    //Return a readable output
    __host__ uint32_t functionsToFilePretty(uint32_t* functionInfo, uint32_t width, uint32_t count) {
        std::ofstream os("outputPretty.txt", std::ofstream::out | std::ofstream::trunc | std::ios::binary);

        os << "Function Count: " << std::to_string(count);
        os << "\nFunction Size: " << std::to_string(width);

        for (uint32_t i = 0; i < count * width; i++) {
            if (!(i % width)) {
                os << "\nFunction " << std::to_string(i / width) << ": ";
            }
            os << "( " << std::to_string(i % width) << " -> " << std::to_string(functionInfo[i]) << "), ";
        }

        os.close();
        return false;
    }
}