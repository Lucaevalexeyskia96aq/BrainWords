#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>
#include "lib/hash/sha3_ver3.cuh"
//#include "lib/hash/sha256.cu"

#include "lib/secp256k1/secp256k1_common.cuh"

#ifdef _DEBUG
#define THREAD_STEPS 3
#else
#define THREAD_STEPS 10
#endif

#ifdef _DEBUG
#define THREAD_STEPS_BRUTE 5
#else
#define THREAD_STEPS_BRUTE 50
#endif

#define PREFIX_MAX_LEN 25
#define SUFFIX_MAX_LEN 20

#define ALPHABET_LEN 11

const char ALPHABET[12] = "0123456789@";
__device__ __constant__ char _ALPHABET[12] = "0123456789@";

cudaError_t loadPrefix(const char* _prefix, size_t const prefixLen);
cudaError_t loadLevel(int _level);
cudaError_t loadIteration(int _i);
cudaError_t loadShaPre(uint32_t* _pre);
cudaError_t cudaMemcpyToSymbol_BLOOM_FILTER(uint32_t* _bloomFilterPtr);
cudaError_t cudaMemcpyToSymbol_BLOOM_FILTER_MASK64(uint64_t* bloomFilterMask);
cudaError_t cudaMemcpyToSymbol_USE_BLOOM_FILTER(uint32_t* useBloomFilter);

cudaError_t loadWindow(unsigned int windowSize, unsigned int windows);

__global__ void worker(bool* isResult, bool* buffResult, const char* __restrict__ lines, const uint32_t* __restrict__ indexes, const uint32_t indexes_size, const secp256k1_ge_storage* __restrict__ precPtr, const size_t precPitch);
__global__ void workerEth(bool* isResult, bool* buffResult, const char* __restrict__ lines, const uint32_t* __restrict__ indexes, const uint32_t indexes_size, const short workType, const bool sha256hash, const secp256k1_ge_storage* __restrict__ precPtr, const size_t precPitch);

__global__ void workerBrute(bool* isResult, uint16_t* buffResult, size_t const prefixLen, uint64_t const startSuffix, const secp256k1_ge_storage* __restrict__ precPtr, const size_t precPitch);
__global__ void workerBruteEth(bool* isResult, uint16_t* buffResult, size_t const prefixLen, uint64_t const startSuffix, const short workType, const bool sha256hash, const secp256k1_ge_storage* __restrict__ precPtr, const size_t precPitch);

__global__ void workerBruteSuffix(bool* isResult, uint16_t* buffResult, size_t const prefixLen, uint64_t const startSuffix, const secp256k1_ge_storage* __restrict__ precPtr, const size_t precPitch);
__global__ void workerBruteEthSuffix(bool* isResult, uint16_t* buffResult, size_t const prefixLen, uint64_t const startSuffix, const short workType, const bool sha256hash, const secp256k1_ge_storage* __restrict__ precPtr, const size_t precPitch);

__global__ void shaPre(size_t const prefixLen, uint32_t* hashPre);



__global__ void ecmult_big_create(secp256k1_gej* gej_temp, secp256k1_fe* z_ratio, secp256k1_ge_storage* precPtr, size_t precPitch, unsigned int bits);
//__global__ void kernelLoadTarget(const unsigned char* _HASH160, size_t _HASH160Len, const unsigned char* _PUB, size_t _PUBLen);

//__device__ void calculateSha256(int* toHash, char* suffixBuffer, uint64_t suffix, size_t prefixLen, beu32* d_hash);
//__device__ void sha256Kernel(beu32* const hash, C16(COMMA, EMPTY));
//__device__ char* ulltoa10(uint64_t value, char* buf);

__device__ bool checkHash(const uint32_t hash[5]);
__device__ bool checkHashEth(unsigned char d_hash[32]);

__device__ __inline__ int lltoa(uint64_t* __restrict__ val, char* __restrict__ buf, const char* __restrict__ dict);

__device__ __inline__ void _GetHashSha256(const char* __restrict__ toHash, size_t const prefixLen, uint32_t* d_hash);
__device__ __inline__ void _GetHashSha256(const char* __restrict__ toHash, size_t const prefixLen, uint32_t* d_hash, const int count);

__device__ __noinline__ void _GetHashSha256FromPre(char* toHash, int& level, uint64_t suffix, size_t const prefixLen, uint32_t* d_hash, uint32_t* hashPre, const char* dict, const int count);
__device__ __inline__ void _GetHashSha256Pre(char* toHash, int level, size_t const prefixLen, uint32_t* d_hash, uint32_t* hashPre);

__device__ __inline__ void _GetHashSha256FromPre(char* toHash, int& level, uint64_t suffix, size_t const prefixLen, uint32_t* d_hash, uint32_t* hashPre, const char* dict);
