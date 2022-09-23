#include "utils.h"

__global__ void gemm(const float *a, const float *b, float *c, const size_t m,
                     const size_t n, const size_t k) {
  int i = blockIdx.x;
  int j = threadIdx.x;
  float acc = 0;
  for (size_t t = 0; t < k; ++t) {
    acc += a[i * 1024 + t] * b[j * 1024 + t];
  }
  c[i * 1024 + j] = acc;
}

int main() {
  const size_t m = 1024;
  const size_t n = 1024;
  const size_t k = 1024;
  const size_t n_iter = 100;

  float *host_a, *host_b, *host_c;
  cudaMallocHost(&host_a, m * k * sizeof(float));
  cudaMallocHost(&host_b, n * k * sizeof(float));
  cudaMallocHost(&host_c, m * n * sizeof(float));
  random_init<float>(host_a, m * k);
  random_init<float>(host_b, n * k);

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, m * k * sizeof(float));
  cudaMalloc(&d_b, n * k * sizeof(float));
  cudaMalloc(&d_c, m * n * sizeof(float));

  cudaMemcpy(d_a, host_a, m * k * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(d_b, host_b, n * k * sizeof(float), cudaMemcpyDefault);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // warmup
  for (size_t i = 0; i < 10; ++i) {
    gemm<<<1024, 1024>>>(d_a, d_b, d_c, m, n, k);
  }

  cudaEventRecord(start);
  for (size_t i = 0; i < n_iter; ++i) {
    gemm<<<1024, 1024>>>(d_a, d_b, d_c, m, n, k);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms;
  cudaEventElapsedTime(&ms, start, end);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  size_t flop = m * n * k * 2;
  double gflops = (double(flop) / 1e9) / ((double(ms) / n_iter) / 1e3);
  printf("GFlops: %f GFlops\n", gflops);

  cudaMemcpy(host_c, d_c, m * n * sizeof(float), cudaMemcpyDefault);

  // TODO: check correctness of host_c

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFreeHost(host_a);
  cudaFreeHost(host_b);
  cudaFreeHost(host_c);
  return 0;
}
