#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;
typedef vector<vector<float>> matrix;

#define CUDA_ERROR_CHECK
#define CUDA_ACCURACY_CHECK

#ifdef CUDA_ERROR_CHECK
#define checkCudaError(val) check((val), #val, __FILE__, __LINE__)
#else
#define checkCudaError(val) ((void)0)
#endif

//Define a macro to make Index straightforward.
#define I(_r, _c) ((_r) * nx + (_c))

void check(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "'\n";
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << "\n";
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void compute_b(float *u, float *v, float *b, int nx, int ny,
		         double dx, double dy, double dt, double rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
      b[I(j, i)] = rho * (1 / dt *
		   ((u[I(j, i + 1)] - u[I(j, i - 1)]) / (2 * dx) + (v[I(j + 1, i)] - v[I(j - 1, i)]) / (2 * dy)) -
	           powf((u[I(j, i + 1)] - u[I(j, i - 1)]) / (2 * dx), 2) -
	           2 * ((u[I(j + 1, i)] - u[I(j - 1, i)]) / (2 * dy) * (v[I(j, i + 1)] - v[I(j, i - 1)]) / (2 * dx)) -
	           powf((v[I(j + 1, i)] - v[I(j - 1, i)]) / (2 * dy), 2));
    }
}

__global__ void update_p(float *p, float *pn, float *b, int nx, int ny, double dx, double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {

      p[I(j, i)] = (pow(dy, 2) * (pn[I(j, i + 1)] + pn[I(j, i - 1)]) +
	            pow(dx, 2) * (pn[I(j + 1, i)] + pn[I(j - 1, i)]) -
	            b[I(j, i)] * pow(dx, 2) * pow(dy, 2)) /
	           (2 * (pow(dx, 2) + pow(dy, 2)));
    }

    // Apply boundary conditions
    if (j < ny) {
      p[I(j, 0)] = p[I(j, 1)];
      p[I(j, nx - 1)] = p[I(j, nx - 2)];
    }

    if (i < nx) {
      p[I(0, i)] = p[I(1, i)];
      p[I(ny-1, i)] = 0;
    }
}

__global__ void compute_uv(float *u, float *v, float *un, float *vn, float *p, int nx, int ny,
		           double dx, double dy, double dt, double rho, double nu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
      u[I(j, i)] = un[I(j, i)] - un[I(j, i)] * dt / dx * (un[I(j, i)] - un[I(j, i - 1)]) -
                   un[I(j, i)] * dt / dy * (un[I(j, i)] - un[I(j - 1, i)]) -
                   dt / (2 * rho * dx) * (p[I(j, i + 1)] - p[I(j, i - 1)]) +
                   nu * dt / (dx * dx) * (un[I(j, i + 1)] - 2 * un[I(j, i)] + un[I(j, i - 1)]) +
                   nu * dt / (dy * dy) * (un[I(j + 1, i)] - 2 * un[I(j, i)] + un[I(j - 1, i)]);

      v[I(j, i)] = vn[I(j, i)] - vn[I(j, i)] * dt / dx * (vn[I(j, i)] - vn[I(j, i - 1)]) -
                   vn[I(j, i)] * dt / dy * (vn[I(j, i)] - vn[I(j - 1, i)]) -
                   dt / (2 * rho * dy) * (p[I(j + 1, i)] - p[I(j - 1, i)]) +
                   nu * dt / (dx * dx) * (vn[I(j, i + 1)] - 2 * vn[I(j, i)] + vn[I(j, i - 1)]) +
                   nu * dt / (dy * dy) * (vn[I(j + 1, i)] - 2 * vn[I(j, i)] + vn[I(j - 1, i)]);
    }

    // Apply boundary conditions
    if (j < ny) {
        u[I(j, 0)] = 0; u[I(j, nx - 1)] = 0;
        v[I(j, 0)] = 0; v[I(j, nx - 1)] = 0;
    }

    if (i < nx) {
        u[I(0, i)] = 0;
	u[I(ny - 1, i)] = 1;
        v[I(0, i)] = 0;
        v[I(ny - 1,i)] = 0;
    }
}

#if defined(CUDA_ACCURACY_CHECK)
__global__ void compute_squared_diff(float *v1, float *v2, float *diff, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    diff[idx] = (v1[idx] - v2[idx]) * (v1[idx] - v2[idx]);
  }
}

__global__ void reduce_sum(float *input, float *output, int size) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  sdata[tid] = (i < size) ? input[i] : 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

float compute_l2_norm(matrix& v1_m, float *v2, int size) {
  float *d_diff, *d_sum, *h_sum;
  int numBlocks = (size + 255) / 256;

  // v1_m is on host. So its contents must be copied to a region that cuda can access.
  float *v1;
  checkCudaError(cudaMallocManaged(&v1, size));
  {
    auto k = 0;
    for (auto const& r : v1_m) {
      for (auto const& c : r) {
        v1[k] = c;
        ++k;
      }
    }
  }

  checkCudaError(cudaMalloc(&d_diff, size * sizeof(float)));
  checkCudaError(cudaMalloc(&d_sum, numBlocks * sizeof(float)));
  h_sum = new float[numBlocks];

  compute_squared_diff<<<numBlocks, 256>>>(v1, v2, d_diff, size);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());

  reduce_sum<<<numBlocks, 256, 256 * sizeof(float)>>>(d_diff, d_sum, size);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());

  checkCudaError(cudaMemcpy(h_sum, d_sum, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

  float sum = 0.0f;
  for (int i = 0; i < numBlocks; ++i) {
    sum += h_sum[i];
  }

  checkCudaError(cudaFree(d_diff));
  checkCudaError(cudaFree(d_sum));
  checkCudaError(cudaFree(v1));
  delete[] h_sum;

  return sqrt(sum);
}
#endif

int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2.0 / (nx - 1);
  double dy = 2.0 / (ny - 1);
  double dt = 0.01;
  double rho = 1.0;
  double nu = 0.02;

  size_t size = nx * ny * sizeof(float);

#if defined(CUDA_ACCURACY_CHECK)
  matrix u(ny,vector<float>(nx));
  matrix v(ny,vector<float>(nx));
  matrix p(ny,vector<float>(nx));
  matrix b(ny,vector<float>(nx));
  matrix un(ny,vector<float>(nx));
  matrix vn(ny,vector<float>(nx));
  matrix pn(ny,vector<float>(nx));
  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      u[j][i] = 0;
      v[j][i] = 0;
      p[j][i] = 0;
      b[j][i] = 0;
    }
  }
#endif

  // Allocate unified memory with prefix d_(cuda Device)
  float *d_u, *d_v, *d_p, *d_b, *d_un, *d_vn, *d_pn;
  checkCudaError(cudaMallocManaged(&d_u, size));
  checkCudaError(cudaMallocManaged(&d_v, size));
  checkCudaError(cudaMallocManaged(&d_p, size));
  checkCudaError(cudaMallocManaged(&d_b, size));
  checkCudaError(cudaMallocManaged(&d_un, size));
  checkCudaError(cudaMallocManaged(&d_vn, size));
  checkCudaError(cudaMallocManaged(&d_pn, size));

  // Initialize u,v, p and b to 0
  checkCudaError(cudaMemset(d_u, 0, size));
  checkCudaError(cudaMemset(d_v, 0, size));
  checkCudaError(cudaMemset(d_p, 0, size));
  checkCudaError(cudaMemset(d_b, 0, size));

  // Define grid and block dimensions
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");

#if defined(CUDA_ACCURACY_CHECK)
  printf("Results by CUDA will be verified.\n"
         "u, v, and p are derived from CPU, whereas d_u, d_v and d_p are from CUDA.\n"
         "||x|| represents L2 norm\n");
#endif


  for (int n = 0; n < nt; ++n) {
    // Compute b
    compute_b<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_b, nx, ny, dx, dy, dt, rho);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    for (int it = 0; it<nit; it++) {
      // Copy p to pn
      checkCudaError(cudaMemcpy(d_pn, d_p, size, cudaMemcpyDeviceToDevice));

      // Update p
      update_p<<<numBlocks, threadsPerBlock>>>(d_p, d_pn, d_b, nx, ny, dx, dy);
      checkCudaError(cudaGetLastError());
      checkCudaError(cudaDeviceSynchronize());
    }

    // Copy u to un, and v to vn, respectively.
    checkCudaError(cudaMemcpy(d_un, d_u, size, cudaMemcpyDeviceToDevice));
    checkCudaError(cudaMemcpy(d_vn, d_v, size, cudaMemcpyDeviceToDevice));

    // Compute u and v
    compute_uv<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_un, d_vn, d_p, nx, ny, dx, dy, dt, rho, nu);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

#if defined(CUDA_ACCURACY_CHECK)
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
        // Compute b[j][i]
        b[j][i] = rho * (1 / dt *
                  ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
                  pow((u[j][i+1] - u[j][i-1]) / (2 * dx), 2) -
		  2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) * (v[j][i+1] - v[j][i-1]) / (2 * dx)) -
		  pow((v[j+1][i] - v[j-1][i]) / (2 * dy), 2));

      }
    }

    for (int it=0; it<nit; it++) {

      for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
	  pn[j][i] = p[j][i];
	}
      }

      for (int j=1; j<ny-1; j++) {
        for (int i=1; i<nx-1; i++) {
	 // Compute p[j][i]
         p[j][i] = (pow(dy,2) * (pn[j][i+1] + pn[j][i-1]) +
                    pow(dx,2) * (pn[j+1][i] + pn[j-1][i]) -
                    b[j][i] * pow(dx,2) * pow(dy,2)) /
                    (2 * (pow(dx,2) + pow(dy,2)));
	}
      }

      for (int j=0; j<ny; j++) {
        // Compute p[j][0] and p[j][nx-1]
        p[j][0] = p[j][1];
        p[j][nx-1] = p[j][nx-2];
      }

      for (int i=0; i<nx; i++) {
	// Compute p[0][i] and p[ny-1][i]
        p[0][i] = p[1][i];
        p[ny-1][i] = 0;
      }
    }

    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        un[j][i] = u[j][i];
	vn[j][i] = v[j][i];
      }
    }

    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
	// Compute u[j][i] and v[j][i]
        u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) -
                  un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) -
                  dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1]) +
                  nu * dt / (dx * dx) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1]) +
                  nu * dt / (dy * dy) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);

        v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) -
                  vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) -
                  dt / (2 * rho * dy) * (p[j+1][i] - p[j-1][i]) +
                  nu * dt / (dx * dx) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1]) +
                  nu * dt / (dy * dy) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
      }
    }

    for (int j=0; j<ny; j++) {
      // Compute u[j][0], u[j][nx-1], v[j][0], v[j][nx-1]
      u[j][0] = 0; u[j][nx-1] = 0;
      v[j][0] = 0; v[j][nx-1] = 0;
    }

    for (int i=0; i<nx; i++) {
      // Compute u[0][i], u[ny-1][i], v[0][i], v[ny-1][i]
      u[0][i] = 0; u[ny-1][i] = 1;
      v[0][i] = 0; v[ny-1][i] = 0;
    }
#endif

    // Write to file every 10th iteration
    if (n % 10 == 0) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          ufile << d_u[I(j, i)] << " ";
          vfile << d_v[I(j, i)] << " ";
          pfile << d_p[I(j, i)] << " ";
        }
      }
      ufile << "\n";
      vfile << "\n";
      pfile << "\n";

#if defined(CUDA_ACCURACY_CHECK)
      float u_norm = compute_l2_norm(u, d_u, nx * ny);
      float v_norm = compute_l2_norm(v, d_v, nx * ny);
      float p_norm = compute_l2_norm(p, d_p, nx * ny);
      printf("n:%3d,||u-d_u||:%g, ||v-d_v||:%g, ||p-d_p||:%g\n",
             n, u_norm, v_norm, p_norm);

#endif
    }
  }

  // Close files
  ufile.close();
  vfile.close();
  pfile.close();

  // Free device memory
  checkCudaError(cudaFree(d_u));
  checkCudaError(cudaFree(d_v));
  checkCudaError(cudaFree(d_p));
  checkCudaError(cudaFree(d_b));
  checkCudaError(cudaFree(d_un));
  checkCudaError(cudaFree(d_vn));
  checkCudaError(cudaFree(d_pn));

  return 0;
}
