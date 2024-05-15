#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm> // For std::max

#define CONFIG_CHKERR

// Error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
#if defined(CONFIG_CHKERR)
  if (code != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
#else
  (void)code;
  (void)file;
  (void)line;
  (void)abort;
#endif
}

__global__ void initializeBuckets(int *bucket, int range) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < range) {
    bucket[idx] = 0;
  }
}

__global__ void countKeys(int *keys, int *bucket, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    atomicAdd(&bucket[keys[idx]], 1);
  }
}

// CUDA kernel to compute prefix sums
__global__ void computePrefixSums(int *bucket, int *prefixSums, int range) {
  extern __shared__ int temp[];
  int tid = threadIdx.x;

  // Load bucket values into shared memory
  if (tid < range) {
    temp[tid] = bucket[tid];
  } else {
    temp[tid] = 0;
  }
  __syncthreads();

  // Perform inclusive scan
  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    int t = 0;
    if (tid >= offset) {
      t = temp[tid - offset];
    }
    __syncthreads();
    temp[tid] += t;
    __syncthreads();
  }

  // Write results to prefixSums array
  if (tid < range) {
    prefixSums[tid] = temp[tid];
  }
}

__global__ void reorderKeys(int *keys, int *bucket, int *prefixSums, int range) {
  int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIdx < range) {
    int start = (globalIdx == 0) ? 0 : prefixSums[globalIdx - 1];
    for (int j = 0; j < bucket[globalIdx]; j++) {
      keys[start + j] = globalIdx;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i = 0; i < n; i++) {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  int *d_key, *d_bucket, *d_prefixSums;
  cudaCheckError(cudaMalloc(&d_key, n * sizeof(int)));
  cudaCheckError(cudaMalloc(&d_bucket, range * sizeof(int)));
  cudaCheckError(cudaMalloc(&d_prefixSums, range * sizeof(int)));

  cudaCheckError(cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemset(d_bucket, 0, range * sizeof(int)));

  const int threadsPerBlock = 256;
  const int blocksPerKey = (n + threadsPerBlock - 1) / threadsPerBlock;
  const int blocksPerRange = (range + threadsPerBlock - 1) / threadsPerBlock;

  initializeBuckets<<<blocksPerRange, threadsPerBlock>>>(d_bucket, range);
  cudaCheckError(cudaGetLastError());

  countKeys<<<blocksPerKey, threadsPerBlock>>>(d_key, d_bucket, n);
  cudaCheckError(cudaGetLastError());

  computePrefixSums<<<1, std::max(threadsPerBlock, range), threadsPerBlock * sizeof(int)>>>(d_bucket, d_prefixSums, range);
  cudaCheckError(cudaGetLastError());

  reorderKeys<<<blocksPerRange, threadsPerBlock>>>(d_key, d_bucket, d_prefixSums, range);
  cudaCheckError(cudaGetLastError());

  cudaCheckError(cudaMemcpy(key.data(), d_key, n * sizeof(int), cudaMemcpyDeviceToHost));

  cudaFree(d_key);
  cudaFree(d_bucket);
  cudaFree(d_prefixSums);

  for (int i=0; i<n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");

  return 0;
}
