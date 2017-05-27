#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <cublas_v2.h>

//#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>
#include <helper_math.h>

__global__ void selectKernel(float*S, int*index, int m, int n, int k) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < m && y < k) {
		S[x + index[x * n + y] * m] = 1;
	}
}

__global__ void subKernel(float *origin, float*sub, int rs, int re, int cs,
		int ce, int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int r = re - rs;
	int c = ce - cs;
	if (x < r && y < c) {
		sub[x + y * r] = origin[x + rs + (y + cs) * m];
	}
}

__global__ void eTimesSPKernel(int*row, int*col, float*val, float*d, int m,
		int n) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < m) {
		int len = row[x + 1] - row[x];
		if (y < len) {
			val[row[x] + y] *= d[x + col[row[x] + y] * m];
		}
	}
}

__device__ __host__ void etimesVecDev(int*col1, int nnz1, float*v1, int*col2,
		int nnz2, float*v2, float* d, int m) {
	int c1, c2;
	int o = -1;
	int i1 = 0;
	int i2 = 0;
	while (i1 < nnz1 && i2 < nnz2) {
		c1 = col1[i1];
		c2 = col2[i2];
		if (c1 == c2) {
			d[c1] = v1[i1++] * v2[i2++];
			o = c1;
		} else if (c1 < c2) {
			for (int i = o + 1; i <= c1; ++i)
				d[i] = 0;
			o = c1;
			i1++;
		} else {
			for (int i = o + 1; i <= c2; ++i)
				d[i] = 0;
			o = c2;
			i2++;
		}
	}
	int nnz = fmaxf(nnz1, nnz2);
	for (int i = nnz; i < m; ++i)
		d[i] = 0;
}

__global__ void etimesKernel(int*row1, int*col1, float*v1, int*row2, int*col2,
		float*v2, float*d, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < m) {
		int nnz1 = row1[x + 1] - row1[x];
		int nnz2 = row2[x + 1] - row2[x];
		etimesVecDev(&col1[row1[x]], nnz1, &v1[row1[x]], &col2[row2[x]], nnz2,
				&v2[row2[x]], &d[x * n], n);
	}
}

__global__ void rowNormSPKernel(int*row, float*val, float*r, int m) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < m) {
		cublasHandle_t handle;
		cublasCreate(&handle);
		int len = row[i + 1] - row[i];
		float norm;
		cublasSnrm2(handle, len, &val[row[i]], 1, &norm);
		if (norm < 1e-10)
			norm = 0;
		else
			norm = 1 / norm;
		r[i] = norm;
		cublasSscal(handle, len, &norm, &val[row[i]], 1);
		cublasDestroy(handle);
	}
}
__global__ void rowMulKernel(int*row, float*val, float alpha, int m) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < m) {
		cublasHandle_t handle;
		cublasCreate(&handle);
		int len = row[i + 1] - row[i];
		float sum;
		cublasSasum(handle, len, &val[row[i]], 1, &sum);
		sum--;
		if (sum > 0)
			sum = powf(sum, -alpha);
		cublasSscal(handle, len, &sum, &val[row[i]], 1);
		cublasDestroy(handle);
	}
}
__global__ void rowSumKernel(int*row, float*val, int m, float*res) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < m) {
		cublasHandle_t handle;
		cublasCreate(&handle);
		int len = row[i + 1] - row[i];
		cublasSasum(handle, len, &val[row[i]], 1, &res[i]);
		cublasDestroy(handle);
	}
}
//__device__ cublasHandle_t handle;

__global__ void colNormKernel(float*r, float*d, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < n) {
		cublasHandle_t handle;
		cublasCreate(&handle);
		float norm;
		cublasSnrm2(handle, m, &d[m * x], 1, &norm);
		r[x] = norm * norm;
		cublasDestroy(handle);
	}
}
__global__ void eTimesKernel(float*va, const float*vb, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		va[i] *= vb[i];
	}
}
__global__ void eDivKernel(float*va, const float*vb, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		va[i] /= max(vb[i], 1e-10);
	}
}
__global__ void signPlusKernel(float*d, float*s, float v, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		float sign = 1.0f;
		if (s[i] < 0)
			sign = -1.0f;
		d[i] += v * sign;
	}
}
__global__ void createIdentityKernel(float *devMatrix, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (y < n && x < n) {
		int index = x * n + y;
		if (x == y)
			devMatrix[index] = 1;
		else
			devMatrix[index] = 0;
	}
}
__global__ void setValueKernel(float* mat, float alpha, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < n)
		mat[x] = alpha;
}
__global__ void toDiagKernel(float*d, const float*v, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < n && y < n) {
		if (x == y)
			d[x * (n + 1)] = v[x];
		else
			d[x * n + y] = 0;
	}
}
__global__ void setZeroDiagKernel(float*d, int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < m) {
		d[x * (m + 1)] = 0;
	}
}
__global__ void diagKernel(float* mat, float*d, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (y < n && x < m && x == y)
		d[x] = mat[x * n + y];
}
__global__ void sqrtKernel(float* v, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < n)
		v[x] = sqrt(v[x]);
}
__global__ void projectKernel(float* v, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < n && v[x] < 0)
		v[x] = 0;
}
__global__ void shrinkKernel(float tau, float* d, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < m && y < n) {
		float v = d[x * n + y];
		float sign = 1.0f;
		if (v < 0) {
			v = -v;
			sign = -1.0f;
		}
		v -= tau;
		if (v < 0)
			v = 0;
		else
			v *= sign;
		d[x * n + y] = v;
	}
}

__global__ void repmatKernel(float*r, float*d, int mm, int mn, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < mm && y < mn) {
		int i = x % m;
		int j = y % n;
		r[x + y * mm] = d[i * n + j];
	}
}

__global__ void predictKernel(int*row1, int*col1, float*v1, int*row2, int*col2,
		float*v2, int*row3, int*col3, float*d, float*tR, int m, int n) {
	int u = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	if (u < m && i < n) {
		float sum = 0;
		int c1, c2;
		int i1 = 0;
		int i2 = 0;
		int nnz1 = row1[u + 1] - row1[u];
		int nnz2 = row2[i + 1] - row2[i];

		bool contain = false;
		for (int j = row3[u]; j < row3[u + 1] && col3[j] <= i; ++j)
			if (col3[j] == i) {
				contain = true;
				break;
			}

		while (i1 < nnz1 && i2 < nnz2) {
			c1 = col1[row1[u] + i1];
			c2 = col2[row2[i] + i2];
			if (c1 == c2) {
				if (contain) //j != i
					sum += d[u + c1 * m] * (v1[row1[u] + i1] - v2[row2[i] + i2])
							* v2[row2[i] + i2];
				else
					sum += d[u + c1 * m] * v1[row1[u] + i1] * v2[row2[i] + i2];
				i1++;
				i2++;
			} else if (c1 < c2)
				i1++;
			else
				i2++;
		}
		tR[u + i * m] = sum;
	}
}

__global__ void subvecKernel(int*row, int*col, int r, float*v, float*d) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int nnz = row[r + 1] - row[r];
	if (x < nnz)
		d[col[row[r] + x]] = v[row[r] + x];
}

__global__ void timesDiagKernel(float*r, float*A, float*d, int m, int n,
		float alpha, bool left) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < m && y < n) {
		if (!left)
			r[x + y * m] = alpha * A[x + y * m] * d[y];
		else
			r[x + y * m] = alpha * A[x + y * m] * d[x];
	}
}

__global__ void getRowKernel(float*v, float*d, int r, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < n)
		d[x] = v[r + x * m];
}

__global__ void setRowKernel(float*v, float*d, int r, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < n)
		v[r + x * m] = d[x];
}

__device__ float bpranking(float*v, int m, int u, int i, int j) {
	float s = v[u + i * m] - v[u + j * m];
	s = expf(-s);
	s = 1 / (1 + s);
	return logf(s);
}

__global__ void lossfuncKernel(float*v, int*row, int*col, int m, int n,
		float*d) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < m) {
		int nnz = row[x + 1] - row[x];
		if (y < nnz) {
			int c = 0;
			d[row[x] + y] = 0;
			for (int i = 0; i < n; ++i) {
				if (i == col[row[x] + c]) {
					c++;
					continue;
				}
				d[row[x] + y] += bpranking(v, m, x, col[row[x] + y], i);
			}
		}
	}
}

__global__ void plusDiagKernel(float*A, float*d, float a, float b, int m,
		int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < m && x < n) {
		A[x * (m + 1)] = a * A[x * (m + 1)] + b * d[x];
	}
}
//__global__ void lossfuncKernel(float*R, float*pR, int m, int n, float*d) {
//	int x = blockDim.x * blockIdx.x + threadIdx.x;
//	int y = blockDim.y * blockIdx.y + threadIdx.y;
//	if (x < m && y < n) {
//		float r1 = R[x + y * m];
//		if (r1 > 0.5) {
//			d[x+y*m] = 0;
//			for (int i = 0; i < n; ++i) {
//				float r2 = R[x + i * m];
//				if (r2 < 0.5) {
//					float s = pR[x + y * m]-pR[x + i * m];
//					s = expf(-s);
//					s = s / (1 + s);
//					d[x+y*m] += logf(s);
//				}
//			}
//		}
//	}
//}
