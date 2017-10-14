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
	double*cu_val = 0;
	thrust::device_ptr<double> val;
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
	void cbind(Dense*r, Dense*d, double a, double b);
	void cbind(Dense*d, double a, double b);
	double dot(Dense*);
	void eDiv(Dense*);
	void eig();
	void eig(Dense*D);
	void eig(Dense*Q, Dense*D);
	void eigs(int k, Dense*Q);
	void svd(Dense*U, Dense*V, Dense*S);
	void eTimes(Dense*d);
	void eTimes(Dense*d, double a);
	void eTimes(Dense*r, Dense*d, double a);
	double frobenius();
	void getCol(Dense*d, int c);
	double getElem(int i, int j);
	void getRow(Dense*d, int r, bool column);
	void getRow(Dense*d, int r);
	void gemv(Dense*x, Dense*y, double alpha, double beta, bool trans);
	void ger(Dense*x, Dense*y, double);
	void keepCol(Dense*r,int*, int n);
	int length() {
		return m * n;
	}
	void ltimes(Dense*A, double alpha, bool trans, bool tA);
	void ltimes(Dense*r, Dense*A, double alpha, bool trans, bool tA);
	void ltimesDiag(Dense*, Dense*, double alpha, bool trans);
	void initial(int m, int n);
	void input(const char*);
	void inv(Dense*);
	void pinv(Dense*, double tol);
	void truncation(int k);
	double norm1();
	double norm2();
	void orth();
	void plus(double, double);
	void plus(Dense*, double, double, bool);
	void plus(Dense*r, Dense*d, double, double, bool, bool);
	void plusDiag(Dense*r, Dense*d, double, double, bool);
	void plusDiag(Dense*d, double, double);
	void plusDiag(double, double);
	void pow(double ind);
	void print();
	void print(const char*);
	void project();
	void rbind(Dense*r, Dense*d, double a, double b);
	void rbind(Dense*d, double a, double b);
	void recip();
	void repmat(Dense*, int, int);
	void rowNorm(Dense*);
	void rowNorm();
	void rowSquare(Dense*);
	void rowSum(Dense*);
	void rowSum();
	void rtimes(Dense*A, double alpha, bool trans, bool tA);
	void rtimes(Dense*r, Dense*A, double alpha, bool trans, bool tA);
	void scale(double);
	void setCol(Dense*d, int c);
	void setDiagValue(double v);
	void setElem(int i, int j, double v);
	void setIdentity();
	void setIdentity(int);
	void setRandom();
	void setRow(Dense*d, int r);
	void setValue(double v);
	void shrink(double tau);
	void signPlus(Dense*, double);
	void solve(Dense*);
//	void sortKeyCol()
	thrust::device_ptr<int> sortKeyCol(bool greater);
	thrust::device_ptr<int> sortKeyRow(bool greater);
	double square();
	void square_root();
	void sub(Dense*, int rs, int re, int cs, int ce);
	void subCol(Dense*, int*, int);
	double sum();
	void timesDiag(Dense*, Dense*, double alpha, bool left, int n);
	void timesDiag(Dense*, Dense*, double alpha, bool left);
	void timesVec(Dense*, Dense*, bool trans);
	void transpose();
	void transpose(Dense*);
	//y=alpha op(A)x+beta y
	//void gemm(Dense*, Dense*, float alpha, float beta, bool tA, bool tB);
	//void times(Dense*, Dense*, bool tA, bool tB);
	void times(Dense*A, Dense*B, double alpha, double beta, bool tA, bool tB);//C=alpha*C+beta*op(A)*op(B)
	void times(Dense*r, double a);
	void times(double a);
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
	double trace(Dense*d);
	~Dense();
};

class Sparse {
public:
//	int*row = 0;
//	int*col = 0;
//	float*val = 0;
	thrust::device_ptr<int> row;
	thrust::device_ptr<int> col;
	thrust::device_ptr<double> val;
	int*cu_row = 0;
	int*cu_row_index = 0;
	int*cu_col = 0;
	int*cu_col_index = 0;
	double*cu_val = 0;
	double*trans_val = 0;
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
	void colNorm();
	void colNorm(Dense*, bool normed);
	void colSum(Dense*);
	void colVec(Dense*d, int c, bool row);
	void colVec(Dense*d, int c);
	void csrmm2(Dense*C, Dense*B, bool, bool, double a, double b);// C=α∗op(A)∗op(B)+β∗C
	void csrmv(Dense*y, Dense*x, double alpha, double beta, bool trans);
	void eTimes(Dense*);
	void eTimes(Dense*, Sparse*);
	double getElem(int i, int j);
	void initialBoth();
	void initialCSC();
	void initialCSR();
	void innerTimes(Sparse*);	//S*St
	void innerTimes(Dense*r, Dense*A);	//B=S*A*St
	void outerTimes(Sparse*);	//St*S
	void outerTimes(Dense*r, Dense*A);	//B=St*A*S
	void inv(Dense*);
	void pinv(Dense*, double tol);
	void plus(Dense*r, double a, double b, bool);  //r=a*op(this)+b*r
	void plus(Sparse*, Sparse*, double, double);
	void pow(double);
	void print();
	void printFull();
	void readCSC(const char*);
	void readCSR(const char*);
	void rowMultiply(double);
	void rowNorm();
	void rowNorm(Dense*, bool normed);
	void rowSum(Dense*);
	void rowVec(Dense*d, int r, bool column);
	void rowVec(Dense*d, int r);
	void selfTimes(Dense*, Dense*);
	void setIdentity(int m);
	void setDiag(Dense*d);
	void times(Dense*r, Dense*d, bool transA, bool transB);
	void times(Sparse*, Sparse*, bool transA, bool transB);
	void timesDiag(Sparse*, Dense*, double a, bool trans);
	void toDense(Dense*);
	void transpose();
	void transpose(Sparse*);
	void updateCSC();
	void updateCSR();
	void uploadCSC(int*, int*, double*);
	void uploadCSR(int*, int*, double*);
	void writeCSC(const char*);
	void writeCSR(const char*);
	~Sparse();
	//C=a*op(A)*op(B)+b*C
	//y=alpha*op(A)*x+beta*y
};

