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

__global__ void selectKernel(double*S, int*index, int m, int n, int k) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < m && y < k) {
		S[x + index[x * n + y] * m] = 1;
	}
}

__global__ void subKernel(double *origin, double*sub, int rs, int re, int cs,
		int ce, int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int r = re - rs;
	int c = ce - cs;
	if (x < r && y < c) {
		sub[x + y * r] = origin[x + rs + (y + cs) * m];
	}
}

__global__ void eTimesSPKernel(int*row, int*col, double*val, double*d, int m,
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

__device__ __host__ void etimesVecDev(int*col1, int nnz1, double*v1, int*col2,
		int nnz2, double*v2, double* d, int m) {
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

__global__ void etimesKernel(int*row1, int*col1, double*v1, int*row2, int*col2,
		double*v2, double*d, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < m) {
		int nnz1 = row1[x + 1] - row1[x];
		int nnz2 = row2[x + 1] - row2[x];
		etimesVecDev(&col1[row1[x]], nnz1, &v1[row1[x]], &col2[row2[x]], nnz2,
				&v2[row2[x]], &d[x * n], n);
	}
}

__global__ void rowNormSPKernel(int*row, double*val, double*r, int m,
		bool normed) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < m) {
		cublasHandle_t handle;
		cublasCreate(&handle);
		int len = row[i + 1] - row[i];
		double norm;
		cublasDnrm2(handle, len, &val[row[i]], 1, &norm);
		r[i] = norm;
		if (norm < 1e-10)
			norm = 0;
		else
			norm = 1 / norm;
		if (normed)
			cublasDscal(handle, len, &norm, &val[row[i]], 1);
		cublasDestroy(handle);
	}
}

__global__ void rowMulKernel(int*row, double*val, double alpha, int m) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < m) {
		cublasHandle_t handle;
		cublasCreate(&handle);
		int len = row[i + 1] - row[i];
		double sum;
		cublasDasum(handle, len, &val[row[i]], 1, &sum);
		sum--;
		if (sum > 0)
			sum = powf(sum, -alpha);
		cublasDscal(handle, len, &sum, &val[row[i]], 1);
		cublasDestroy(handle);
	}
}
__global__ void rowSumKernel(int*row, double*val, int m, double*res) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < m) {
		cublasHandle_t handle;
		cublasCreate(&handle);
		int len = row[i + 1] - row[i];
		cublasDasum(handle, len, &val[row[i]], 1, &res[i]);
		cublasDestroy(handle);
	}
}
//__device__ cublasHandle_t handle;

__global__ void colNormKernel(double*r, double*d, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < n) {
		cublasHandle_t handle;
		cublasCreate(&handle);
		double norm;
		cublasDnrm2(handle, m, &d[m * x], 1, &norm);
		r[x] = norm * norm;
		cublasDestroy(handle);
	}
}
__global__ void eTimesKernel(double*va, const double*vb, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		va[i] *= vb[i];
	}
}
__global__ void eDivKernel(double*va, const double*vb, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		va[i] /= max(vb[i], 1e-10);
	}
}
__global__ void signPlusKernel(double*d, double*s, double v, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		double sign = 1.0f;
		if (s[i] < 0)
			sign = -1.0f;
		d[i] += v * sign;
	}
}
__global__ void createIdentityKernel(double *devMatrix, int n) {
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
__global__ void setValueKernel(double* mat, double alpha, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < n)
		mat[x] = alpha;
}
__global__ void toDiagKernel(double*d, const double*v, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < n && y < n) {
		if (x == y)
			d[x * (n + 1)] = v[x];
		else
			d[x * n + y] = 0;
	}
}
__global__ void setZeroDiagKernel(double*d, int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < m) {
		d[x * (m + 1)] = 0;
	}
}
__global__ void diagKernel(double* mat, double*d, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (y < n && x < m && x == y)
		d[x] = mat[x * n + y];
}
__global__ void sqrtKernel(double* v, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < n)
		v[x] = sqrt(v[x]);
}
__global__ void projectKernel(double* v, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < n && v[x] < 0)
		v[x] = 0;
}
__global__ void shrinkKernel(double tau, double* d, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < m && y < n) {
		double v = d[x * n + y];
		double sign = 1.0f;
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

__global__ void repmatKernel(double*r, double*d, int mm, int mn, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < mm && y < mn) {
		int i = x % m;
		int j = y % n;
		r[x + y * mm] = d[i * n + j];
	}
}

__global__ void predictKernel(int*row1, int*col1, double*v1, int*row2, int*col2,
		double*v2, int*row3, int*col3, double*d, double*tR, int m, int n) {
	int u = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	if (u < m && i < n) {
		double sum = 0;
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

__global__ void subvecKernel(int*row, int*col, int r, double*v, double*d) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int nnz = row[r + 1] - row[r];
	if (x < nnz)
		d[col[row[r] + x]] = v[row[r] + x];
}

__global__ void timesDiagKernel(double*r, double*A, double*d, int m, int n,
		int k, double alpha, bool left) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (k >= n) {
		if (x < m && y < n) {
			if (!left)
				r[x + y * m] = alpha * A[x + y * m] * d[y];
			else
				r[x + y * m] = alpha * A[x + y * m] * d[x];
		}
	} else {
		if (x < m && y < k) {
			if (!left)
				r[x + y * m] = alpha * A[x + y * m] * d[y];
			else
				r[x + y * m] = alpha * A[x + y * m] * d[x];
		}
	}
}

__global__ void timesDiagSpKernel(int*row, int*col, double*v, double*d, int m,
		double alpha) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < m) {
		int nnz = row[x + 1] - row[x];
		if (y < nnz) {
			int c = col[row[x] + y];
			v[row[x] + y] *= alpha * d[c];
		}
	}
}

__global__ void getRowKernel(double*v, double*d, int r, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < n)
		d[x] = v[r + x * m];
}

__global__ void setRowKernel(double*v, double*d, int r, int m, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < n)
		v[r + x * m] = d[x];
}

__device__ double bpranking(double*v, int m, int u, int i, int j) {
	double s = v[u + i * m] - v[u + j * m];
	s = expf(-s);
	s = 1 / (1 + s);
	return logf(s);
}

__global__ void lossfuncKernel(double*v, int*row, int*col, int m, int n,
		double*d) {
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

__global__ void plusDiagKernel(double*A, double*d, double a, double b, int m,
		int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < m && x < n) {
		A[x * (m + 1)] = a * A[x * (m + 1)] + b * d[x];
	}
}

__global__ void rbindKernel(double*r, double*d1, double*d2, int m1, int m2,
		int n, double a, double b) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (y < n) {
		int m = m1 + m2;
		if (x < m1)
			r[x + m * y] = a * d1[x + m1 * y];
		else if (x < m1 + m2)
			r[x + m * y] = b * d2[x - m1 + m2 * y];
	}
}

__global__ void setkKernel(int*order, double*v, int m, int n, int k) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < m && y < k)
		v[order[x * n + y] * m + x] = 1;
}

//__global__ void lossfuncKernel(double*R, double*pR, int m, int n, double*d) {
//	int x = blockDim.x * blockIdx.x + threadIdx.x;
//	int y = blockDim.y * blockIdx.y + threadIdx.y;
//	if (x < m && y < n) {
//		double r1 = R[x + y * m];
//		if (r1 > 0.5) {
//			d[x+y*m] = 0;
//			for (int i = 0; i < n; ++i) {
//				double r2 = R[x + i * m];
//				if (r2 < 0.5) {
//					double s = pR[x + y * m]-pR[x + i * m];
//					s = expf(-s);
//					s = s / (1 + s);
//					d[x+y*m] += logf(s);
//				}
//			}
//		}
//	}
//}
