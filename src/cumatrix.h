#ifndef CUMATRIX_H_
#define CUMATRIX_H_

#endif

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

#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
//#include <thrust/iterator/transform_iterator.h>

#include <set>
#include <algorithm>
#include <curand.h>
#include <fstream>
#define THREADS 512
#define PER_THREADS 16
#define FOR(i,start,end) for(int i=start;i<(end);i++)
#define PRINT_MATRIX(v,m,n) for(int i=0;i<m;i++) { \
	                        for(int j=0;j<n;j++)   \
							std::cout<<v[i + j*m] << " ";\
							std::cout<<std::endl;\
							}
#define PRINT_VECTOR(v,n) for (int i=0;i<(n);i++) std::cout<<v[i]<<" ";

class Dense {
public:
	float*cu_val = 0;
	thrust::device_ptr<float> val;
	int m, n;
public:
public:
	static cublasHandle_t handle;
	static cusolverDnHandle_t solver_handle;
	void clean();
	void clone(Dense*);
	void colNorm(Dense*);
	void colSquare(Dense*);
	void colSum(Dense*);
	void copyto(Dense*);
	void diag(Dense*);
	float dot(Dense*);
	void eDiv(Dense*);
	void eig();
	void eig(Dense*D);
	void eig(Dense*Q, Dense*D);
	void eigs(int k, Dense*Q);
	void eTimes(Dense*);
	void eTimes(Dense*, float a);
	float frobenius();
	void getCol(Dense*d, int c);
	float getElem(int i, int j);
	void getRow(Dense*d, int r);
	void gemv(Dense*x, Dense*y, float alpha, float beta, bool trans);
	void ger(Dense*x, Dense*y, float);
public:
	int length() {
		return m * n;
	}
	void ltimes(Dense*A, float alpha, bool trans, bool tA);
	void ltimes(Dense*r, Dense*A, float alpha, bool trans, bool tA);
	void ltimesDiag(Dense*, Dense*, float alpha, bool trans);
	void initial(int m, int n);
	void input(const char*);
	void inv();
	float norm1();
	float norm2();
	void orth();
	void plus(float, float);
	void plus(Dense*, float, float, bool);
	void plus(Dense*r, Dense*d, float, float, bool, bool);
	void pow(float ind);
	void print();
	void print(const char*);
	void project();
	void repmat(Dense*, int, int);
	void rowNorm(Dense*);
	void rowSquare(Dense*);
	void rowSum(Dense*);
	void rtimes(Dense*A, float alpha, bool trans, bool tA);
	void rtimes(Dense*r, Dense*A, float alpha, bool trans, bool tA);
	void scale(float);
	void setCol(Dense*d, int c);
	void setDiagValue(float v);
	void setElem(int i, int j, float v);
	void setIdentity();
	void setIdentity(int);
	void setRandom();
	void setRow(Dense*d, int r);
	void setValue(float v);
	void shrink(float tau);
	void signPlus(Dense*, float);
	void solve(Dense*x, Dense*b);
//	void sortKeyCol()
	thrust::device_ptr<int> sortKeyCol(bool greater);
	thrust::device_ptr<int> sortKeyRow(bool greater);
	float square();
	void square_root();
	void sub(Dense*, int rs, int re, int cs, int ce);
	float sum();
	void timesDiag(Dense*, Dense*, float alpha, bool left);
	void timesVec(Dense*, Dense*, bool trans);
	void transpose();
	void transpose(Dense*);
	//y=alpha op(A)x+beta y
	//void gemm(Dense*, Dense*, float alpha, float beta, bool tA, bool tB);
	//void times(Dense*, Dense*, bool tA, bool tB);
	void times(Dense*A, Dense*B, float alpha, float beta, bool tA, bool tB);//C=alpha*C+beta*op(A)*op(B)
	void times(Dense*r, float a);
	void times(float a);
	//B=alpha*op(B)*op(A)
	//r=alpha*op(B)*op(A)
	//B=alpha*op(A)*op(B)
	//r=alpha*op(A)*op(B)
	//void plus(float);
	//	void plus(Dense* addend, float);
//	void plus(Dense* addend, float, float);
//	void plus(Dense* res, Dense* addend, float, float);
	//d=alpha*d+beta*ones
	//this=alpha*this+beta*op(d)
	//	void plusDiag(Dense*)
	float trace(Dense*d);
	~Dense();
};

class Sparse {
public:
//	int*row = 0;
//	int*col = 0;
//	float*val = 0;
	thrust::device_ptr<int> row;
	thrust::device_ptr<int> col;
	thrust::device_ptr<float> val;
	int*cu_row = 0;
	int*cu_row_index = 0;
	int*cu_col = 0;
	int*cu_col_index = 0;
	float*cu_val = 0;
	float*trans_val = 0;
	int m = 0;
	int n = 0;
	int nnz = 0;
public:
	static cusparseMatDescr_t descr;
	static cusparseMatDescr_t descr_L;
	static cusparseMatDescr_t descr_U;
	static cusparseHandle_t handle;
public:
	void clean();
	void clone(Sparse*);
	void colSum(Dense*);
	void colVec(Dense*d, int c);
	void csrmm2(Dense*C, Dense*B, bool, bool, float a, float b);
	void csrmv(Dense*y, Dense*x, float alpha, float beta, bool trans);
	void diagTimes(Sparse*, Dense*, bool trans);
	void eTimes(Dense*);
	void eTimes(Dense*, Sparse*);
	float getElem(int i, int j);
	void initialBoth();
	void initialCSC();
	void initialCSR();
	void innerTimes(Sparse*);
	void inv(Dense*);
	void outerTimes(Sparse*);
	void plus(Dense*r, float, float, bool);
	void plus(Sparse*, Sparse*, float, float);
	void print();
	void printFull();
	void readCSC(const char*);
	void readCSR(const char*);
	void rowMultiply(float);
	void rowNorm();
	void rowNorm(Dense*);
	void rowSum(Dense*);
	void rowVec(Dense*d, int r);
	void selfTimes(Dense*, Dense*);
	void setIdentity(int m);
	void setDiag(Dense*d);
	void times(Dense*, Dense*, bool transA, bool transB);
	void times(Sparse*, Sparse*, bool transA, bool transB);
	void toDense(Dense*);
	void transpose();
	void uploadCSC(int*, int*, float*);
	void uploadCSR(int*, int*, float*);
	void writeCSC(const char*);
	void writeCSR(const char*);
	~Sparse();
	//C=a*op(A)*op(B)+b*C
	//y=alpha*op(A)*x+beta*y
};

