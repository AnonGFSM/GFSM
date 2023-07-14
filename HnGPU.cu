#pragma once
#include "HnGPU.cuh"
#include <chrono>

void cudaExit(cudaError_t error) {
    info_printf("\nCuda Error: %s", cudaGetErrorString(error));
    csv_printf("%s,", cudaGetErrorString(error));
    exit(error);
}

namespace HnGPU {

    __device__ uint32_t debugSet = 0;

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

    __host__ __device__ int32_t getBlockCount(int globalRange, int localRange) {
        return (globalRange / localRange) + 1;
    }

    __device__ void updateHeader(uint32_t* solutionHeader) {
        solutionHeader[HNFUNC_OFFSET] = atomicAdd(&solutionHeader[HNFUNC_NEWOFFSET], 0);
        solutionHeader[HNFUNC_COUNT] = atomicAdd(&solutionHeader[HNFUNC_NEWCOUNT], 0);
    }


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

#ifdef GPUDEBUG
            //if (ix < solutionHeader[HNFUNC_COUNT] - 1) {
            //    uint32_t dif = functionDegrees[ix];
            //    if (dif != vSize && ix < 100) {
            //        printf("\nWrong Degree %i %i", dif, vSize);
            //    }
            //}
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

            if (fns == 0) {
                //printf("\nFn Found %lu", fn.mapping);
            }

            if (ix == solutionHeader[HNFUNC_COUNT] - 1) {
                solutionHeader[HNFUNC_NEWOFFSET] = solutionHeader[HNFUNC_OFFSET] + solutionHeader[HNFUNC_COUNT] + batchOffset;
                solutionHeader[HNFUNC_NEWCOUNT] = functAllocs[ix] + vSize;
            }

            functCounts[ix] = vSize;
        }

    }


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


            if (fn->mapping == -1) {
                debug_printf("\nBad Thread Found! %u", ix);
                __trap();
            }


            if (nextFirstOccurence == 1) {
                bBfrFunctionDegrees[ix] = codomainEdges[fn->mapping];
            }

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



            fn = cFn;

            /*
            if (ix == 37) {
                printf("\n %i", fn->mapping);
                for (uint32_t i2 = 0; codomainEdges[fn->mapping + i2]; i2++) {
                    printf(" {%i}", codomainEdges[fn->mapping + i2]);
                }
            }*/

            uint32_t foundLocs = numOfReqs;

            //Remember to flip ordering of preprocess (IT IS FLIPPED COMPARED TO ORIGINAL!)

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

    __global__ void hnRem(uint32_t* bfr, uint32_t val, uint32_t count) {
        uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
        if (ix < count) {
            bfr[ix] = bfr[ix] - val;
        }
    }

   
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
                //printf("\nNew Func Count %i -> %i (%i)", solutionHeader[HNFUNC_COUNT], functOffsets[ix] + (fn.mapping != -1), solutionHeader[HNFUNC_NEWCOUNT]);
            }
        }

    }

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

    __device__ void dumpSoln(HnFunction* functions, uint32_t count) {
        printf("\nDumping Buffer");
        for (uint32_t i = 0; i < count; i++) {
            printf("{%i, %i}", functions[i].mapping, functions[i].previous);
        }
    }

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

        //HnGen

        uint32_t blockCount = getBlockCount(solutionHeader[HNFUNC_COUNT], HBLOCKSIZE);

        hnGen << < blockCount, HBLOCKSIZE >> > (codomainEdges, functions, bBfrFunctions, functionDegrees, scanBfr, solutionHeader, functCounts,
            maxValency, requirementHeader[HNREQS_H_FIRSTQVERTEX + reqIndex], batchOffset);

        if (cudaSuccess != cudaDeviceSynchronize()) return true;

        updateHeader(solutionHeader);

        debug_printf("\nHnGen -> %u", solutionHeader[HNFUNC_COUNT]);

        blockCount = getBlockCount(solutionHeader[HNFUNC_COUNT], HBLOCKSIZE);

        //HnCheck

        hnCheck << < blockCount, HBLOCKSIZE >> > (codomainEdges, functions, bBfrFunctions, bBfrFunctionDegrees, solutionHeader, functCounts, scanBfr,
            requirements, requirementHeader[HNREQS_H_NUM + reqIndex], requirementHeader[HNREQS_H_INDEX + reqIndex], depth,
            requirementHeader[HNREQS_H_FIRSTQVERTEX + nextReqIndex]);

        if (cudaSuccess != cudaDeviceSynchronize()) return true;


        //HnRemove

        debug_printf("\nHnCheck -> %u", solutionHeader[HNFUNC_COUNT]);

        if (cudaSuccess != hnScan(scanBfr, scanBfrTemp, SCAN_LIMIT * sizeof(int), functCounts, solutionHeader, SCAN_LIMIT)) return true;

        //printf("\nHnScan");

        blockCount = getBlockCount(solutionHeader[HNFUNC_COUNT], HBLOCKSIZE);

        if (batchOffset) {
            uint32_t funcCount = solutionHeader[HNFUNC_COUNT];
            if (scanBfr[funcCount - 1] + functCounts[funcCount - 1] + solutionHeader[HNFUNC_OFFSET] + batchOffset >= SOLN_LIMIT) {
                debug_printf("\nBatch Overflow scan: %u, count: %u, offset: %u, batchOffset: %u", scanBfr[funcCount - 1], functCounts[funcCount - 1], solutionHeader[HNFUNC_OFFSET], batchOffset);
                return true;
            }
        }

        if (depth == 0) {
            //return false;
        }

        hnRemove << < blockCount, HBLOCKSIZE >> > (functions, bBfrFunctions, functionDegrees, bBfrFunctionDegrees, solutionHeader, scanBfr, batchOffset);

        if (cudaSuccess != cudaDeviceSynchronize()) return true;

        updateHeader(solutionHeader);

        debug_printf("\nHnRemove -> %u", solutionHeader[HNFUNC_COUNT]);

        pcsv_printf("HnStep: %u (%u),", solutionHeader[HNFUNC_COUNT], depth);

        //if (cudaSuccess != hnSort(scanBfr, scanBfrTemp, MAXSCANSIZE * sizeof(int), functionDegrees, scanBfr, solutionHeader)) return true;

        if (cudaSuccess != hnScan(scanBfr, scanBfrTemp, SCAN_LIMIT * sizeof(int), functionDegrees, solutionHeader, SCAN_LIMIT)) return true;

        //printf("\nHnScan");

        return false;
    }

    __device__ uint32_t divCeil(uint32_t l, uint32_t r) {
        return (l / r) + ((l / r) != 0);
    }

#define BATCHINGSPACE 1024 * 16

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

//HnInit
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

        for (int depth = 0; depth < depthTarget; depth++) {

            uint32_t funcCount = solutionHeader[HNFUNC_COUNT];
            uint32_t predFuncCount = scanBfr[funcCount - 1] + functionDegrees[funcCount-1];
            uint32_t funcOffset = solutionHeader[HNFUNC_OFFSET];
            uint32_t predCeil = predFuncCount + funcCount + funcOffset;
            debug_printf("\nHnPred -> %u", predFuncCount);
            pcsv_printf("HnPred: %u (%u),", predFuncCount, depth);
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
                        pSize = batchSize;
                        if (pSize > funcCount - pF) {
                            pSize = funcCount - pF;
                        }

                        //printf("\npSize %lu pC %lu", pSize, parentCount);

                        //Safety
                        const uint32_t safety = 4;
                        //debug_printf("\nFuncs %lu", newFuncCount);
                        for (uint32_t i = 0; i < safety; i++) {
                            //printf("\n%lu Compare %lu to %lu", scanBfr[pF + pSize - 1], scanBfr[pF + pSize - 1] - scanBfr[pF], space - newFuncCount);
                            if (scanBfr[pF + pSize - 1] == 0) {
                                //printf("\nBROKEN Compare, %lu, %lu", scanBfr[pF + pSize - 1], scanBfr[pF]);
                                return;
                            }
                            if (scanBfr[pF+ pSize-1] - scanBfr[pF] > space - newFuncCount) {
                                pSize /= 2;
                            }
                            else {
                                break;
                            }

                            if (i == safety - 1 || pSize == 0) {
                                //printf("\nBatch too large");
                                return;
                            }
                        }


                        uint32_t expectedDif = scanBfr[pF + pSize - 1] - scanBfr[pF];

                        //printf("\npSize %lu degTailSize %lu (%lu) expectedDif %lu", pSize, scanBfr[pF + pSize - 1] - scanBfr[pF], scanBfr[pF + pSize - 1] + functionDegrees[pF + pSize - 1] - scanBfr[pF], expectedDif);

                        solutionHeader[HNFUNC_COUNT] = pSize;


                        cudaMemcpyAsync(scanBfrBack, scanBfr + pF, pSize * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
                        if (filteredCandidates) {
                            blockCount = getBlockCount(solutionHeader[HNFUNC_COUNT], HBLOCKSIZE);
                            hnRem << <blockCount, HBLOCKSIZE >> > (scanBfrBack, filteredCandidates, solutionHeader[HNFUNC_COUNT]);
                            hnDeg << <blockCount, HBLOCKSIZE >> > (functions, codomainEdges, functionDegrees, solutionHeader, depth-1, requirementHeader[HNREQS_H_FIRSTQVERTEX + (depth+1)*3]);
                        }
                        if(pF + pSize < funcCount)
                            filteredCandidates = scanBfr[pF + pSize];
                        cudaDeviceSynchronize();

                        if (scanBfrBack[pSize - 1] != expectedDif) {
                            //printf("\nCorrupted Data %lu != %lu", scanBfrBack[pSize - 1], expectedDif);
                            for (int i = 0; i < 8; i++) {
                                //printf("\n%lu,", scanBfrBack[pSize - 1 - i]);
                            }
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
                        //printf("\nNew Funcs: %lu", solutionHeader[HNFUNC_COUNT]);
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
                        debug_printf("General loop failure %i", depth);
                        csv_printf("Loop failure %i,", depth);
                        return;
                }
            }

            

            if (volatile int count = solutionHeader[HNFUNC_COUNT] == 0) {
                return;
            }

            if (depth == 0) {
                //return;
            }
        }

    }

    __global__ void hnWrite(CCSRSegment* codomainEdges, HnFunction* functions, uint32_t* functionInfo,
                        uint32_t* inverseDataVerticeLocs, int32_t* queryMappingData,
                        CCSR::StaggerInfo* staggersInfoGPU,
                        uint32_t functionCount, uint32_t functionOffset,
                        uint32_t querySize) {
        uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t* fInfo = &functionInfo[ix * querySize];


        //return;
        if (ix < functionCount) {
            HnFunction function = functions[ix + functionOffset];
            for (uint32_t i = 0; i < querySize; i++) {
                uint32_t mapping = function.mapping;

                uint32_t degree = codomainEdges[mapping];

                CCSR::StaggerInfo staggerInfoGPU = staggersInfoGPU[degree - 1];

                uint32_t normalisedIndex = staggerInfoGPU.normOffset + (mapping - staggerInfoGPU.offset) / (degree + 1);

                fInfo[(uint32_t)queryMappingData[i]] = inverseDataVerticeLocs[normalisedIndex];

                function = functions[function.previous];
            }

        }

    }

}

namespace HnSetup {

    __host__ std::vector<HnSolution>* solve(const CCSR::CCSRGraph& ccsrQuery, const CCSR::CCSRGraph& ccsrData,
        const CCSR::CCSRStagger& queryStagger, const CCSR::CCSRStagger& dataStagger) {

        auto start = std::chrono::steady_clock::now();

        uint32_t* requirements, * requirementHeader;

        //print(ccsrQuery);
        int32_t* mappingData;

        uint32_t maxValencyQuery = preProcessQuery(ccsrQuery, &requirements, &requirementHeader, &mappingData);

        uint32_t maxValencyData = CCSR::getMaxValency(ccsrData);


        HnFunction* solutions;
        uint32_t* solutionHeader;


        //printf("\nTest2 %i", dataStagger.len);

        //for (int i = 0; i < dataStagger.len; i++) {
            //printf("\nTesting %i", dataStagger.staggers[i]);
        //}

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

        //free(queryMappings);
        //free(dataMappings);
        //free(solutions);
        //free(ccsrQuery.segments);
        //free(ccsrData.segments);
        //free(requirements);
        //free(requirementHeader);
    }

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
        
        //Set Vertex 0 to seen
        (*mappingData)[loc] = 0;
        //printf("\nTest ");
        //for (int i = 0; i < query.count; i++) {
        //    printf("%i, ", (*mappingData)[i]);
        //}

        for (int depth = 0; depth < query.count; depth++) {

            CCSR::genereateCCSRRequirements(query, *requirements, *requirementHeader, maxValency, depth, *mappingData);
            //printf("\nTest ");
            //for (int i = 0; i < query.count; i++) {
            //    printf("%i, ", (*mappingData)[i]);
            //}
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

        /*
        printf("\n");
        for (int i = 0; i < query.count; i++) {
            printf("%i, ", query.invVerticeLocs[i]);
        }

        printf("\n");
        for (int i = 0; i < query.count; i++) {
        	printf("%i, ", (*mappingData)[i]);
        }

        printf("\n");
        for (int i = 0; i < query.count; i++) {
            printf("%i, ", orderingData[i]);
        }

        for (int i = 0; i < query.count; i++) {
            tempMappingData[i] = query.invVerticeLocs[orderingData[i]];
        }*/

        memcpy(*mappingData, tempMappingData, sizeof(uint32_t) * query.count);

        //printf("\n");
        //for (int i = 0; i < query.count; i++) {
        //    printf("%i, ", (*mappingData)[i]);
        //}

        delete [] tempMappingData;
        delete [] orderingData;

        //(*requirements)[4] = 1;
        //(*requirements)[8] = 1;
        //(*requirements)[12] = 1;
        //(*requirements)[16] = 1;

        //(*requirementHeader)[11] = 1;
        //(*requirementHeader)[14] = 1;

        /* Requirement Header Usage
        * requirementHeader[0] = Number Of Requirements
        * requirementHeader[1] = Requirements Index
        * requirementHeader[2] = First Query Vertex Depth
        */
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
                //printf("\nWe broke!");
                pcsv_printf("Precompiler Failure: Query Gen,", 0);
                exit(1);
            }
        }

        return maxValency;
    }

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

    __global__ void dummy() {

    }

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

        for (uint32_t i = 0; i < 9; i++) {
            //printf("\nTest %lu %lu", dataStagger.staggersInfo[i].offset, dataStagger.staggersInfo[i].normOffset);
        }

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

        /*
        cudaEvent_t start, stop;

        float elapsedTime = 0.0f;

        auto startC = std::chrono::steady_clock::now();

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);*/

        //Launch Solve

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

#ifdef FINGERPRINTING        
        std::string frontBfrSum = checkSumCuda(frontBfr + (*solutionHeaderCPU)[HNFUNC_OFFSET], sizeof(HnFunction) * (*solutionHeaderCPU)[HNFUNC_COUNT]);
        std::string backBfrSum = checkSumCuda(backBfr + (*solutionHeaderCPU)[HNFUNC_OFFSET], sizeof(HnFunction) * (*solutionHeaderCPU)[HNFUNC_COUNT]);
        std::string scanBfrSum = checkSumCuda(scanBfr, sizeof(uint32_t) * (*solutionHeaderCPU)[HNFUNC_COUNT]);
        std::string fullBfrSum = checkSumCuda(frontBfr, sizeof(HnFunction) * ((*solutionHeaderCPU)[HNFUNC_COUNT] + (*solutionHeaderCPU)[HNFUNC_OFFSET]));

        debug_printf("\nSolution, Offset: %u, Count: %u", (*solutionHeaderCPU)[HNFUNC_OFFSET], (*solutionHeaderCPU)[HNFUNC_COUNT]);
        debug_printf("\nFrontBfr: %s", frontBfrSum.c_str());
        debug_printf("\nBackBfr: %s", backBfrSum.c_str());
        debug_printf("\nScanBfr: %s", scanBfrSum.c_str());
        debug_printf("\nFullFrontBfr: %s", fullBfrSum.c_str());
#endif

        if ((*solutionHeaderCPU)[HNFUNC_COUNT] == DEBUGCOUNT) { // Memory Limit Failure
            //csv_printf("%s", "Debug State,");
            if (MEM_LIMIT == MEMLIMIT) {
                csv_printf("%s", "Reallocating,");
                info_printf("%s", "\nReallocating");
                //cudaStatus = cudaDeviceReset();
                //if (cudaSuccess != cudaStatus) exit(cudaStatus);

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
                //csv_printf("%f,", elapsed_secondsR.count());
            }
            return;
        }
        else {
            if (MEM_LIMIT == MEMLIMIT) {
                csv_printf("%s", "n/a,n/a,");
            }
        }
        /*
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "\nTime Taken: " << elapsedTime << "ms" << std::endl;
        */
        
        info_printf("\nFunction Count: %i", (*solutionHeaderCPU)[HNFUNC_COUNT]);
        csv_printf("%i,", (*solutionHeaderCPU)[HNFUNC_COUNT]);


        //uint32_t memcpySize = (*solutionHeaderCPU)[HNFUNC_COUNT] + (*solutionHeaderCPU)[HNFUNC_OFFSET];

        //cudaMemcpy(*solutionCPU, frontBfr, sizeof(HnFunction) * memcpySize, cudaMemcpyDeviceToHost);
        //dumpFunctions(*solutionCPU, memcpySize);

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

        HnGPU::hnWrite << <blockCount, HBLOCKSIZE >> > (dataGPU, frontBfr, functionInfoGPU,
            invDataVerticeLocsGPU, queryMappingDataGPU,
            staggersInfoGPU,
            (*solutionHeaderCPU)[HNFUNC_COUNT], (*solutionHeaderCPU)[HNFUNC_OFFSET],
            query.count);

        cudaStatus = cudaDeviceSynchronize();
        if (cudaSuccess != cudaStatus) cudaExit(cudaStatus);

        info_printf("\nhnWrite Completed", 0);

        uint32_t* functionInfo = (uint32_t*)malloc((*solutionHeaderCPU)[HNFUNC_COUNT] * query.count * sizeof(uint32_t));
        //MappingPair* functionInfo;
        //cudaHostAlloc(&functionInfo, (*solutionHeaderCPU)[HNFUNC_COUNT] * query.count * sizeof(MappingPair), cudaHostAllocPortable);
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

        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds5 = end - start;
        info_printf("\nSolution Creation End Time: %fs\n", elapsed_seconds5.count());
        csv_printf("%fs,", elapsed_seconds5.count());


        //printFunctions(functionInfo, query.count, (*solutionHeaderCPU)[HNFUNC_COUNT]);
        //functionsToFile(functionInfo, query.count, (*solutionHeaderCPU)[HNFUNC_COUNT]);
        //functionsToFilePretty(functionInfo, query.count, (*solutionHeaderCPU)[HNFUNC_COUNT]);
    }

    __host__ void printFunctions(uint32_t* functionInfo, uint32_t width, uint32_t count) {
        printf("\nFunctions");

        for (uint32_t i = 0; i < count * width; ) {
            printf("\nFunction %lu: ", i / width);
            for (uint32_t i2 = 0; i2 < width; i2++, i++) {
                printf("(%lu, %lu), ", i2, functionInfo[i]);
            }
        }
    }
    
    
    __host__ uint32_t functionsToFile(uint32_t* functionInfo, uint32_t width, uint32_t count) {
        std::ofstream os("output.fs", std::ofstream::out | std::ofstream::trunc | std::ios::binary);

        os.write((char*) &count, sizeof(uint32_t));
        os.write((char*) &width, sizeof(uint32_t));
        os.write((char*) functionInfo, count * width * sizeof(uint32_t));

        os.close();

        return false;
    }

    /*
    __host__ uint32_t functionsToFile(uint32_t* functionInfo, uint32_t width, uint32_t count) {
        std::ofstream os("output2.fs", std::ofstream::out | std::ofstream::trunc | std::ios::binary);

        std::filebuf* inbuf = os.rdbuf();
        char* buf = new char[sizeof(uint32_t) * (2 + count * width)];
        inbuf->pubsetbuf(buf, sizeof(uint32_t) * (2 + count * width));

        inbuf->sputn((char*)&count, sizeof(uint32_t));
        inbuf->sputn((char*)&width, sizeof(uint32_t));
        inbuf->sputn((char*)functionInfo, count * width * sizeof(uint32_t));
        //uint32_t* intBuf = (uint32_t*)buf;
        //intBuf[0] = count;
        //intBuf[1] = width;

        //std::memcpy(intBuf + 2, functionInfo, count * width * sizeof(uint32_t));

        //delete[] buf;

        //inbuf->pubseekoff(sizeof(uint32_t) * (2 + count * width), os.end);
        //inbuf->pubsync();
        inbuf->close();

        return false;
    }*/

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

//Author: https://github.com/System-Glitch/SHA256


SHA256::SHA256() : m_blocklen(0), m_bitlen(0) {
    m_state[0] = 0x6a09e667;
    m_state[1] = 0xbb67ae85;
    m_state[2] = 0x3c6ef372;
    m_state[3] = 0xa54ff53a;
    m_state[4] = 0x510e527f;
    m_state[5] = 0x9b05688c;
    m_state[6] = 0x1f83d9ab;
    m_state[7] = 0x5be0cd19;
}

void SHA256::update(const uint8_t* data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        m_data[m_blocklen++] = data[i];
        if (m_blocklen == 64) {
            transform();

            // End of the block
            m_bitlen += 512;
            m_blocklen = 0;
        }
    }
}

void SHA256::update(const std::string& data) {
    update(reinterpret_cast<const uint8_t*> (data.c_str()), data.size());
}

uint8_t* SHA256::digest() {
    uint8_t* hash = new uint8_t[32];

    pad();
    revert(hash);

    return hash;
}

uint32_t SHA256::rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

uint32_t SHA256::choose(uint32_t e, uint32_t f, uint32_t g) {
    return (e & f) ^ (~e & g);
}

uint32_t SHA256::majority(uint32_t a, uint32_t b, uint32_t c) {
    return (a & (b | c)) | (b & c);
}

uint32_t SHA256::sig0(uint32_t x) {
    return SHA256::rotr(x, 7) ^ SHA256::rotr(x, 18) ^ (x >> 3);
}

uint32_t SHA256::sig1(uint32_t x) {
    return SHA256::rotr(x, 17) ^ SHA256::rotr(x, 19) ^ (x >> 10);
}


static constexpr std::array<uint32_t, 64> K = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

void SHA256::transform() {
    uint32_t maj, xorA, ch, xorE, sum, newA, newE, m[64];
    uint32_t state[8];

    for (uint8_t i = 0, j = 0; i < 16; i++, j += 4) { // Split data in 32 bit blocks for the 16 first words
        m[i] = (m_data[j] << 24) | (m_data[j + 1] << 16) | (m_data[j + 2] << 8) | (m_data[j + 3]);
    }

    for (uint8_t k = 16; k < 64; k++) { // Remaining 48 blocks
        m[k] = SHA256::sig1(m[k - 2]) + m[k - 7] + SHA256::sig0(m[k - 15]) + m[k - 16];
    }

    for (uint8_t i = 0; i < 8; i++) {
        state[i] = m_state[i];
    }

    for (uint8_t i = 0; i < 64; i++) {
        maj = SHA256::majority(state[0], state[1], state[2]);
        xorA = SHA256::rotr(state[0], 2) ^ SHA256::rotr(state[0], 13) ^ SHA256::rotr(state[0], 22);

        ch = choose(state[4], state[5], state[6]);

        xorE = SHA256::rotr(state[4], 6) ^ SHA256::rotr(state[4], 11) ^ SHA256::rotr(state[4], 25);

        sum = m[i] + K[i] + state[7] + ch + xorE;
        newA = xorA + maj + sum;
        newE = state[3] + sum;

        state[7] = state[6];
        state[6] = state[5];
        state[5] = state[4];
        state[4] = newE;
        state[3] = state[2];
        state[2] = state[1];
        state[1] = state[0];
        state[0] = newA;
    }

    for (uint8_t i = 0; i < 8; i++) {
        m_state[i] += state[i];
    }
}

void SHA256::pad() {

    uint64_t i = m_blocklen;
    uint8_t end = m_blocklen < 56 ? 56 : 64;

    m_data[i++] = 0x80; // Append a bit 1
    while (i < end) {
        m_data[i++] = 0x00; // Pad with zeros
    }

    if (m_blocklen >= 56) {
        transform();
        memset(m_data, 0, 56);
    }

    // Append to the padding the total message's length in bits and transform.
    m_bitlen += m_blocklen * 8;
    m_data[63] = m_bitlen;
    m_data[62] = m_bitlen >> 8;
    m_data[61] = m_bitlen >> 16;
    m_data[60] = m_bitlen >> 24;
    m_data[59] = m_bitlen >> 32;
    m_data[58] = m_bitlen >> 40;
    m_data[57] = m_bitlen >> 48;
    m_data[56] = m_bitlen >> 56;
    transform();
}

void SHA256::revert(uint8_t* hash) {
    // SHA uses big endian byte ordering
    // Revert all bytes
    for (uint8_t i = 0; i < 4; i++) {
        for (uint8_t j = 0; j < 8; j++) {
            hash[i + (j * 4)] = (m_state[j] >> (24 - i * 8)) & 0x000000ff;
        }
    }
}

std::string SHA256::toString(const uint8_t* digest) {
    std::stringstream s;
    s << std::setfill('0') << std::hex;

    for (uint8_t i = 0; i < 32; i++) {
        s << std::setw(2) << (unsigned int)digest[i];
    }

    return s.str();
}

std::string checkSumCuda(void* cudaBfr, size_t size) {
#ifdef FINGERPRINTING
    char* temp = new char[size] {};
    cudaError_t error = cudaMemcpy(temp, cudaBfr, size, cudaMemcpyDeviceToHost);
    if (error) {
        cudaExit(error);
    }
    SHA256 sha;
    sha.update((uint8_t*)temp, size);
    uint8_t* digest = sha.digest();
    std::string soln = SHA256::toString(digest);
    delete[] digest;
    delete[] temp;
    return soln;
#else
    return std::string("null");
#endif
}
