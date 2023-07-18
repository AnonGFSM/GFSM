#ifndef EnvH
#define EnvH
#include <stdint.h>
#include "cuda_runtime.h"

#define CCSRRelSize 8
#define CCSRVertSize (32 - CCSRRelSize)
#define MAXSOLNSIZE 2*500'000'000
#define MAXSCANSIZE 2*500'000'000
#define MEMLIMIT 22'000'000'000
#define UPPERMEMLIMIT 44'000'000'000
#define HBLOCKSIZE 256

#define DEBUGCOUNT 1'777'777'777

#define ISOSOLVE
#define SHORTREL
#define GPUDEBUG
#define BATCHING
#define OVERFLOWCHECK
#define PREDICTCHECK
//#define FINGERPRINTING

//#define QUERYDATAPRINT
//#define NAMEDATA

//#define GPUPRINT
#ifdef GPUPRINT
#define debug_printf(f_, ...) printf((f_), __VA_ARGS__)
#else
#define debug_printf(f_, ...) do {} while(0)
#endif

//#define INFOPRINT
#ifdef INFOPRINT
#define info_printf(f_, ...) printf((f_), __VA_ARGS__)
#else
#define info_printf(f_, ...) do {} while(0)
#endif

#define CSVPRINT
#ifdef CSVPRINT
#define csv_printf(f_, ...) printf((f_), __VA_ARGS__)
#else
#define csv_printf(f_, ...) do {} while(0)
#endif

//#define PCSVPRINT
#ifdef PCSVPRINT
#define pcsv_printf(f_, ...) printf((f_), __VA_ARGS__)
#else
#define pcsv_printf(f_, ...) do {} while(0)
#endif

#define HNDEBUG

#if (CUDART_VERSION < 11030)
template<class T>
__host__ cudaError_t cudaMallocAsync(T** ptr, size_t size, cudaStream_t stream) {
    return cudaMalloc(ptr, size);
}

template<class T>
__host__ cudaError_t cudaFreeAsync(T* ptr, cudaStream_t stream) {
    return cudaFree(ptr);
}
#endif

constexpr __device__ __host__ uint32_t CCSRRelation(uint32_t c) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < c; i++) {
        result |= (1 << (31 - i));
    }
    return result;
}

constexpr __device__ __host__ uint32_t CCSRIndex(uint32_t c) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < c; i++) {
        result |= (1 << i);
    }
    return result;
}


#endif
