#include "cumatrix.h"
#include <limits>

double ONE = 1.0f;
double NEG_ONE = -1.0f;
double HALF = 0.5f;
double NEG_HALF = -0.5f;
double ZERO = 0.0f;
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
printf("Error at %s:%d\n",__FILE__,__LINE__);\
exit(EXIT_FAILURE);}} while(0) 

__global__ void subKernel(double *origin, double*sub, int rs, int re, int cs,
		int ce, int m);
__global__ void eTimesSPKernel(int*row, int*col, double*val, double*d, int m,
		int n);
__global__ void etimesKernel(int*row1, int*col1, double*v1, int*row2, int*col2,
		double*v2, double*d, int m, int n);
__global__ void rowNormSPKernel(int*row, double*val, double*r, int m);
__global__ void rowMulKernel(int*row, double*val, double alpha, int m);
__global__ void rowSumKernel(int*row, double*val, int m, double*res);
//__global__ void eTimesKernel(double*va, const double*vb, int n);
//__global__ void eDivKernel(double*va, const double*vb, int n);
__global__ void signPlusKernel(double*d, double*s, double v, int n);
__global__ void timesDiagKernel(double*r, double*A, double*d, int m, int n,
		int k, double alpha, bool trans);
//__global__ void sqrtKernel(double* v, int n);
__global__ void projectKernel(double* v, int n);
__global__ void shrinkKernel(double tau, double* d, int m, int n);
__global__ void repmatKernel(double*r, double*d, int mm, int mn, int m, int n);
__global__ void subvecKernel(int*row, int*col, int r, double*v, double*d);
__global__ void getRowKernel(double*v, double*d, int r, int m, int n);
__global__ void setRowKernel(double*v, double*d, int r, int m, int n);
__global__ void plusDiagKernel(double*A, double*d, double a, double b, int m,
		int n);
__global__ void rbindKernel(double*r, double*d1, double*d2, int m1, int m2,
		int n, double a, double b);

void Dense::clean() {
	if (cu_val)
		checkCudaErrors(cudaFree(cu_val));
	cu_val = 0;
}

void Dense::clone(Dense*d) {
	d->clean();
	d->initial(m, n);
	int len = length();
	checkCudaErrors(cublasDcopy(handle, len, cu_val, 1, d->cu_val, 1));
	//cudaMemcpy(d->cu_val, cu_val, sizeof(float)*m*n, cudaMemcpyDeviceToDevice);
}

void Dense::copyto(Dense* d) {
	int len = length();
	checkCudaErrors(cublasDcopy(handle, len, cu_val, 1, d->cu_val, 1));
}

double Dense::dot(Dense* d) {
	int n = length();
	double res;
	checkCudaErrors(cublasDdot(handle, n, cu_val, 1, d->cu_val, 1, &res));
	return res;
}

void Dense::ger(Dense*x, Dense*y, double a) {
	checkCudaErrors(
			cublasDger(handle, m, n, &a, x->cu_val, 1, y->cu_val, 1, cu_val, m));
}

void Dense::initial(int m, int n) {
	this->m = m;
	this->n = n;
	//printf("m=%d,n=%d\n", m, n);
	checkCudaErrors(cudaMalloc((void** ) &cu_val, sizeof(double) * m * n));
	val = thrust::device_pointer_cast(cu_val);
}

void Dense::input(const char* filename) {
	FILE* file = fopen(filename, "r");
	fscanf(file, "%d %d", &m, &n);
	size_t size = sizeof(double) * m * n;
	double*elem = (double*) malloc(size);
	//int len = m*n;
	int i, j;
	for (i = 0; i < m; ++i)
		for (j = 0; j < n; ++j)
			fscanf(file, "%lf", &elem[i + j * m]);
	//PRINT_MATRIX(val,m,n);
	checkCudaErrors(cudaMalloc((void** ) &cu_val, sizeof(double) * m * n));
	checkCudaErrors(cudaMemcpy(cu_val, elem, size, cudaMemcpyHostToDevice));
	val = thrust::device_pointer_cast(cu_val);
	fclose(file);
	free(elem);
}

void Dense::setRandom() {
	curandGenerator_t gen;
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 123ULL));
	CURAND_CALL(curandGenerateUniformDouble(gen, cu_val, length()));
	CURAND_CALL(curandDestroyGenerator(gen));
}

void Dense::setValue(double v) {
	thrust::fill(val, val + m * n, v);
}

void Dense::setDiagValue(double v) {
	int size = fminf(m, n);
	for (int i = 0; i < size; ++i)
		val[i + i * m] = v;
}

void Dense::setIdentity(int s) {
	initial(s, s);
	setValue(0.0f);
	for (int i = 0; i < s; ++i)
		val[i * s + i] = 1;
}

void Dense::setIdentity() {
	setValue(0.0f);
	setDiagValue(1.0f);
}

void Dense::setElem(int i, int j, double v) {
	val[i + j * m] = v;
}

double Dense::getElem(int i, int j) {
	return val[i + j * m];
}

void Dense::transpose(Dense* trans) {
	int m = this->n;
	int n = this->m;
	trans->clean();
	trans->initial(m, n);
	//checkCudaErrors(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
	checkCudaErrors(
			cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &ONE, cu_val,
					this->m, &ZERO, cu_val, this->m, trans->cu_val, trans->m));
}

void Dense::transpose() {
	Dense* trans = new Dense;
	transpose(trans);
	clean();
	trans->clone(this);
	delete trans;
}

void Dense::colSum(Dense* vec) {
	Dense* tran = new Dense;
	transpose(tran);
	Dense*v = new Dense;
	tran->rowSum(v);
	v->transpose(vec);
	delete tran;
	delete v;
}

void Dense::rowSum(Dense* vec) {
	Dense* one = new Dense;
	one->initial(n, 1);
	one->setValue(1.0f);
	vec->clean();
	vec->initial(m, 1);
	checkCudaErrors(
			cublasDgemv(handle, CUBLAS_OP_N, m, n, &ONE, cu_val, m, one->cu_val, 1, &ZERO, vec->cu_val, 1));
	delete one;
}

void Dense::rowSquare(Dense*r) {
	Dense*m = new Dense;
	this->clone(m);
	m->eTimes(m);
	m->rowSum(r);
	delete m;
}
void Dense::colSquare(Dense*r) {
	Dense*m = new Dense;
	this->clone(m);
	m->eTimes(m);
	m->colSum(r);
	delete m;
}

void Dense::rowNorm(Dense*r) {
	rowSquare(r);
	r->square_root();
}
void Dense::colNorm(Dense*r) {
	colSquare(r);
	r->square_root();
}

void Dense::eig(Dense*Q, Dense*D) {
	D->clean();
	D->initial(m, 1);
	clone(Q);
	int lwork;
	checkCudaErrors(
			cusolverDnDsyevd_bufferSize(solver_handle, CUSOLVER_EIG_MODE_VECTOR,
					CUBLAS_FILL_MODE_UPPER, m, Q->cu_val, m, D->cu_val,
					&lwork));
//	printf("lwork=%d\n", lwork);
	double*work = NULL;
	int* devInfo = NULL;
	checkCudaErrors(cudaMalloc((void** ) &work, sizeof(double) * lwork));
	checkCudaErrors(cudaMalloc((void** ) &devInfo, sizeof(int)));
	cusolverDnDsyevd(solver_handle, CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_UPPER, m, Q->cu_val, m, D->cu_val, work, lwork,
			devInfo);
//	printf("devInfo=%d\n", devInfo);
	if (work)
		checkCudaErrors(cudaFree(work));
	if (devInfo)
		checkCudaErrors(cudaFree(devInfo));
}

void Dense::eig(Dense*D) {
	D->clean();
	D->initial(m, 1);
	int lwork;
	checkCudaErrors(
			cusolverDnDsyevd_bufferSize(solver_handle, CUSOLVER_EIG_MODE_VECTOR,
					CUBLAS_FILL_MODE_UPPER, m, cu_val, m, D->cu_val, &lwork));
//	printf("lwork=%d\n", lwork);
	double*work = NULL;
	int* devInfo = NULL;
	checkCudaErrors(cudaMalloc((void** ) &work, sizeof(double) * lwork));
	checkCudaErrors(cudaMalloc((void** ) &devInfo, sizeof(int)));
	cusolverDnDsyevd(solver_handle, CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_UPPER, m, cu_val, m, D->cu_val, work, lwork,
			devInfo);
//	printf("devInfo=%d\n", devInfo);
	if (work)
		checkCudaErrors(cudaFree(work));
	if (devInfo)
		checkCudaErrors(cudaFree(devInfo));
}

void Dense::eig() {
	double*d = NULL;
	double*work = NULL;
	int* devInfo = NULL;
	checkCudaErrors(cudaMalloc((void** ) &d, sizeof(double) * m));
	int lwork;
	checkCudaErrors(
			cusolverDnDsyevd_bufferSize(solver_handle, CUSOLVER_EIG_MODE_VECTOR,
					CUBLAS_FILL_MODE_UPPER, m, cu_val, m, d, &lwork));
	checkCudaErrors(cudaMalloc((void** ) &work, sizeof(double) * lwork));
	checkCudaErrors(cudaMalloc((void** ) &devInfo, sizeof(int)));
	cusolverDnDsyevd(solver_handle, CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_UPPER, m, cu_val, m, d, work, lwork, devInfo);
	if (work)
		checkCudaErrors(cudaFree(work));
	if (devInfo)
		checkCudaErrors(cudaFree(devInfo));
	if (d)
		checkCudaErrors(cudaFree(d));
}

void Dense::eigs(int k, Dense*Q) {
	Q->clean();
	Q->initial(m, k);
	Q->setRandom();
	Q->orth();
	for (int i = 0; i < 10; ++i) {
		Q->ltimes(this, 1.0f, false, false);
		Q->orth();
	}
}

void Dense::orth() {
	int lwork_geqrf;
	int lwork_orgqr;
	int lwork;
	int*devInfo = NULL;
	double*work = NULL;
	double*tau = NULL;
	checkCudaErrors(cudaMalloc((void** )&tau, sizeof(double) * n));
	checkCudaErrors(cudaMalloc((void** )&devInfo, sizeof(int)));
	checkCudaErrors(
			cusolverDnDgeqrf_bufferSize(solver_handle, m, n, cu_val, m,
					&lwork_geqrf));
	checkCudaErrors(
			cusolverDnDorgqr_bufferSize(solver_handle, m, n, n, cu_val, m, tau,
					&lwork_orgqr));
	lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;
	checkCudaErrors(cudaMalloc((void** )&work, sizeof(double) * lwork));
	checkCudaErrors(
			cusolverDnDgeqrf(solver_handle, m, n, cu_val, m, tau, work, lwork,
					devInfo));
	checkCudaErrors(
			cusolverDnDorgqr(solver_handle, m, n, n, cu_val, m, tau, work,
					lwork, devInfo));
	if (devInfo)
		checkCudaErrors(cudaFree(devInfo));
	if (work)
		checkCudaErrors(cudaFree(work));
	if (tau)
		checkCudaErrors(cudaFree(tau));
}

void Dense::inv(Dense*B) {
	if (m != n) {
		printf("not square\n");
		return;
	}
	B->clean();
	B->initial(m, m);
	B->setIdentity();
	solve(B);
}

struct reciprocal: public thrust::unary_function<double, double> {
	const double a;
	reciprocal(double _a) :
			a(_a) {
	}
	__host__ __device__
	double operator()(const double &x) const {
		if (x > a)
			return 1 / x;
		else
			return 0;
	}
};

void Dense::pinv(Dense*B, double tol) {
	Dense*U = new Dense;
	Dense*V = new Dense;
	Dense*S = new Dense;

	svd(U, V, S);
	thrust::transform(S->val, S->val + S->length(), S->val, reciprocal(tol));
	V->timesDiag(B, S, 1.0, false, U->m);
	B->rtimes(U, 1.0, false, true);
	delete U;
	delete V;
	delete S;
}

void Dense::truncation(int k) {
	if (k > m || k > n) {
		printf("k is larger than m or n\n");
		return;
	}
	Dense*U = new Dense;
	Dense*V = new Dense;
	Dense*S = new Dense;
	svd(U, V, S);
	thrust::fill(S->val + k, S->val + S->length(), 0);
	U->timesDiag(this, S, 1.0, false, V->m);
	rtimes(V, 1.0, false, true);
	delete U;
	delete V;
	delete S;
//	S->sor
}

void Dense::svd(Dense*U, Dense*V, Dense*S) {
	Dense*T = new Dense;
	if (m < n) {
		transpose(T);
		T->svd(V, U, S);
//		U->transpose();
		delete T;
		return;
	}
	clone(T);
	U->clean();
	U->initial(m, m);
	V->clean();
	V->initial(n, n);
	S->clean();
	S->initial(m, 1);
	S->setValue(0);
	int lwork = 0;
	checkCudaErrors(cusolverDnDgesvd_bufferSize(solver_handle, m, n, &lwork));
	double*work = 0;
	double*rwork = 0;
	int*devInfo = 0;
	checkCudaErrors(cudaMalloc((void** )&devInfo, sizeof(int)));
	checkCudaErrors(cudaMalloc((void** )&work, sizeof(double) * lwork));
	checkCudaErrors(
			cusolverDnDgesvd(solver_handle, 'A', 'A', m, n, T->cu_val, m,
					S->cu_val, U->cu_val, m, V->cu_val, n, work, lwork, rwork,
					devInfo));
	V->transpose();
	delete T;
}

void Dense::print() {
	printf("%d %d\n", m, n);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
//			printf("%.2f ", v[i * n + j]);
			std::cout << val[i + j * m] << " ";
		std::cout << std::endl;
	}
}

void Dense::print(const char* filename) {
	std::ofstream out(filename);
	out.precision(std::numeric_limits<double>::max_digits10);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
			out << val[i + j * m] << " ";
		out << std::endl;
	}
	out.close();
}

void Dense::gemv(Dense*x, Dense*y, double alpha, double beta, bool trans) {
	if (trans)
		checkCudaErrors(
				cublasDgemv(handle, CUBLAS_OP_T, m, n, &alpha, cu_val, m, x->cu_val, 1, &beta, y->cu_val, 1));
	else
		checkCudaErrors(
				cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha, cu_val, m, x->cu_val, 1, &beta, y->cu_val, 1));
}

void Dense::timesVec(Dense*r, Dense*v, bool trans) {
	r->clean();
	if (trans)
		r->initial(n, 1);
	else
		r->initial(m, 1);
	gemv(v, r, 1.0f, 0.0f, trans);
}

void Dense::timesDiag(Dense*r, Dense*d, double alpha, bool left, int n) {
	r->clean();
	if (left)
		r->initial(this->n, n);
	else
		r->initial(this->m, n);
	r->setValue(0);
	dim3 threadsPerBlock(PER_THREADS, PER_THREADS);
	dim3 numBlocks((r->m + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(r->n + threadsPerBlock.y - 1) / threadsPerBlock.y);
	timesDiagKernel<<<numBlocks, threadsPerBlock>>>(r->cu_val, cu_val,
			d->cu_val, r->m, r->n, left ? this->m : this->n, alpha, left);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Dense::timesDiag(Dense*r, Dense*d, double alpha, bool left) {
	int n = left ? this->m : this->n;
	timesDiag(r, d, alpha, left, n);
//	r->clean();
//	r->initial(m, n);
//	dim3 threadsPerBlock(PER_THREADS, PER_THREADS);
//	dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
//			(n + threadsPerBlock.y - 1) / threadsPerBlock.y);
//	timesDiagKernel<<<numBlocks, threadsPerBlock>>>(r->cu_val, cu_val,
//			d->cu_val, m, n, alpha, left);
//	checkCudaErrors(cudaDeviceSynchronize());
}

void Dense::times(Dense*A, Dense*B, double alpha, double beta, bool tA,
		bool tB) {
	int k;
	if (tA) {
		k = A->m;
		if (tB)
			checkCudaErrors(
					cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &beta, A->cu_val, A->m, B->cu_val, B->m, &alpha, cu_val, m));
		else
			checkCudaErrors(
					cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &beta, A->cu_val, A->m, B->cu_val, B->m, &alpha, cu_val, m));
	} else {
		k = A->n;
		if (tB)
			checkCudaErrors(
					cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &beta, A->cu_val, A->m, B->cu_val, B->m, &alpha, cu_val, m));
		else
			checkCudaErrors(
					cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &beta, A->cu_val, A->m, B->cu_val, B->m, &alpha, cu_val, m));
	}
}
struct multiply_const: public thrust::unary_function<double, double> {
	const double a;
	multiply_const(double _a) :
			a(_a) {
	}
	__host__ __device__
	double operator()(const double &x) const {
		return x * a;
	}
};
void Dense::times(Dense*r, double a) {
	r->clean();
	r->initial(m, n);
	thrust::transform(val, val + length(), r->val, multiply_const(a));
}
void Dense::times(double a) {
	thrust::transform(val, val + length(), val, multiply_const(a));
}
struct multiply: public thrust::binary_function<double, double, double> {
	const double a;
	multiply(double _a) :
			a(_a) {
	}
	__host__ __device__
	double operator()(const double &x, const double &y) const {
		return x * y * a;
	}
};
void Dense::eTimes(Dense* d, double a) {
	thrust::transform(val, val + length(), d->val, val, multiply(a));
}

void Dense::eTimes(Dense*d) {
	eTimes(d, 1.0f);
}

void Dense::eTimes(Dense*r, Dense*d, double a) {
	r->clean();
	r->initial(m, n);
	thrust::transform(val, val + length(), d->val, r->val, multiply(a));
}

struct divide: public thrust::binary_function<double, double, double> {
	__host__ __device__
	double operator()(const double &x, const double &y) const {
		return x / y;
	}
};
void Dense::eDiv(Dense*d) {
	thrust::transform(val, val + length(), d->val, val, divide());
//	int thread = THREADS;
//	int len = length();
//	int block = (len + thread - 1) / thread;
//	eDivKernel<<<block, thread>>>(cu_val, d->cu_val, len);
//	checkCudaErrors(cudaDeviceSynchronize());
}

void Dense::rtimes(Dense*r, Dense*A, double alpha, bool trans, bool tA) {
	r->clean();
	int m, n;
	m = (trans) ? this->n : this->m;
	n = (tA) ? A->m : A->n;
	r->initial(m, n);
	r->times(this, A, 0.0f, alpha, trans, tA);
}

void Dense::rtimes(Dense*A, double alpha, bool trans, bool tA) {
	Dense*B = new Dense;
	this->clone(B);
	B->rtimes(this, A, alpha, trans, tA);
	delete B;
}

void Dense::ltimes(Dense*r, Dense*A, double alpha, bool trans, bool tA) {
	r->clean();
	int m, n;
	m = (tA) ? A->n : A->m;
	n = (trans) ? this->m : this->n;
	r->initial(m, n);
	r->times(A, this, 0.0f, alpha, tA, trans);
}

void Dense::ltimes(Dense*A, double alpha, bool trans, bool tA) {
	Dense*B = new Dense;
	this->clone(B);
	B->ltimes(this, A, alpha, trans, tA);
	delete B;
}
//void Dense::times(Dense*r, Dense* d, bool tA, bool tB) {
////	int m = this->m;
////	int k = this->n;
////	int n = d->n;
//	r->clean();
//	int m, n, k;
//	r->clean();
//	if (tA) {
//		m = this->n;
//		k = this->m;
//		if (tB) {
//			n = d->m;
//			r->initial(m, n);
//			checkCudaErrors(
//					cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &ONE, cu_val, this->m, d->cu_val, d->m, &ZERO, r->cu_val, m));
//		} else {
//			n = d->n;
//			r->initial(m, n);
//			checkCudaErrors(
//					cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &ONE, cu_val, this->m, d->cu_val, d->m, &ZERO, r->cu_val, m));
//		}
//	} else {
//		m = this->m;
//		k = this->n;
//		if (tB) {
//			n = d->m;
//			r->initial(m, n);
//			checkCudaErrors(
//					cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &ONE, cu_val, this->m, d->cu_val, d->m, &ZERO, r->cu_val, m));
//		} else {
//			n = d->n;
//			r->initial(m, n);
//			checkCudaErrors(
//					cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &ONE, cu_val, this->m, d->cu_val, d->m, &ZERO, r->cu_val, m));
//		}
//	}
//}

//void Dense::plus(Dense*addend, float a) {
//	Dense*r = new Dense;
//	plus(r, addend, 1.0f, a);
//	r->clone(this);
//	delete r;
//}
//
//void Dense::plus(Dense*addend, float a, float b) {
//	Dense*r = new Dense;
//	plus(r, addend, a, b);
//	r->clone(this);
//	delete r;
//}
//
//void Dense::plus(Dense* res, Dense* addend, float a, float b) {
//	plus(res, addend, a, b, false, false);
//}

void Dense::plus(double alpha, double beta) {
	Dense* ones = new Dense;
	ones->initial(m, n);
	ones->setValue(1.0f);
	plus(ones, alpha, beta, false);
	delete ones;
}

void Dense::plus(Dense* d, double a, double b, bool trans) {
	if (trans)
		checkCudaErrors(
				cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, &a, cu_val,
						m, &b, d->cu_val, d->m, cu_val, m));
	else
		checkCudaErrors(
				cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &a, cu_val,
						m, &b, d->cu_val, d->m, cu_val, m));
}

void Dense::plus(Dense* res, Dense* d, double a, double b, bool tA, bool tB) {
	res->clean();
	int m, n;
	if (tA) {
		m = this->n;
		n = this->m;
		res->initial(m, n);
		if (tB)
			checkCudaErrors(
					cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &a,
							cu_val, this->m, &b, d->cu_val, d->m, res->cu_val,
							m));
		else
			checkCudaErrors(
					cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &a,
							cu_val, this->m, &b, d->cu_val, d->m, res->cu_val,
							m));
	} else {
		m = this->m;
		n = this->n;
		res->initial(m, n);
		if (tB)
			checkCudaErrors(
					cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, &a,
							cu_val, this->m, &b, d->cu_val, d->m, res->cu_val,
							m));
		else
			checkCudaErrors(
					cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &a,
							cu_val, this->m, &b, d->cu_val, d->m, res->cu_val,
							m));
	}
}

void Dense::plusDiag(Dense*r, Dense*d, double a, double b, bool tran) {
	if (!tran)
		clone(r);
	else
		transpose(r);
	r->plusDiag(d, a, b);
}

void Dense::plusDiag(Dense*d, double a, double b) {
	int thread = THREADS;
	int block = (m + thread - 1) / thread;
	plusDiagKernel<<<block, thread>>>(cu_val, d->cu_val, a, b, m, n);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Dense::plusDiag(double a, double b) {
	Dense*I = new Dense;
	I->initial(m, 1);
	I->setValue(1.0f);
	plusDiag(I, a, b);
	delete I;
}

void Dense::solve(Dense*x) {
	int size = 0;
	double* buffer = 0;
	double* tau = 0;
	int*info = 0;
	int h_info = 0;
	double one = 1.0f;
	checkCudaErrors(cudaMalloc((void** ) &info, sizeof(int)));
	checkCudaErrors(cudaMalloc((void** ) &tau, sizeof(double) * m));
	checkCudaErrors(
			cusolverDnDgeqrf_bufferSize(solver_handle, m, m, cu_val, m, &size));
	checkCudaErrors(cudaMalloc((void** ) &buffer, sizeof(double) * size));
	checkCudaErrors(
			cusolverDnDgeqrf(solver_handle, m, m, cu_val, m, tau, buffer, size,
					info));
	checkCudaErrors(
			cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));
	if (0 != h_info) {
		fprintf(stderr, "Error: LU factorization failed\n");
	}
//	x->initial(b->m, b->n);
//	checkCudaErrors(
//			cudaMemcpy(x->cu_val, b->cu_val, sizeof(double) * b->length(),
//					cudaMemcpyDeviceToDevice));
	checkCudaErrors(
			cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, x->m,
					x->n, m, cu_val, m, tau, x->cu_val, x->m, buffer, size,
					info));
	checkCudaErrors(
			cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, x->m, x->n, &one, cu_val, m, x->cu_val, x->m));
	checkCudaErrors(cudaFree(buffer));
	checkCudaErrors(cudaFree(tau));
	checkCudaErrors(cudaFree(info));
}

void Dense::scale(double a) {
	int len = length();
	checkCudaErrors(cublasDscal(handle, len, &a, cu_val, 1));
}

double Dense::trace(Dense*d) {
	Dense* r = new Dense;
	transpose(r);
	r->eTimes(d);
	Dense* v = new Dense;
	r->rowSum(v);
	double sum = v->sum();
	delete r;
	delete v;
	return sum;
}

double Dense::sum() {
	Dense* vec = new Dense;
	rowSum(vec);
	Dense* ones = new Dense;
	ones->initial(m, 1);
	ones->setValue(1.0f);
	double s = vec->dot(ones);
	delete vec;
	delete ones;
	return s;
}

double Dense::frobenius() {
	double n = norm2();
	return n * n;
}

double Dense::norm2() {
	double n2;
//cublasSdot(handle, m, cu_val, 1, cu_val, 1, &n2);
	checkCudaErrors(cublasDnrm2(handle, length(), cu_val, 1, &n2));
	return n2;
}

double Dense::norm1() {
	double n1;
	checkCudaErrors(cublasDasum(handle, length(), cu_val, 1, &n1));
	return n1;
}

double Dense::square() {
	double n = norm2();
	return n * n;
}

void Dense::getCol(Dense*d, int c) {
	d->clean();
	d->initial(m, 1);
	checkCudaErrors(
			cudaMemcpy(d->cu_val, &cu_val[c * m], sizeof(double) * m,
					cudaMemcpyDeviceToDevice));
}

void Dense::setCol(Dense*d, int c) {
	checkCudaErrors(
			cudaMemcpy(&cu_val[c * m], d->cu_val, sizeof(double) * m,
					cudaMemcpyDeviceToDevice));
}

thrust::device_ptr<int> Dense::sortKeyCol(bool greater) {
	int len = m * n;
	thrust::device_ptr<int> order = thrust::device_malloc<int>(len);
	thrust::device_vector<int> d(len);
	thrust::device_vector<int> seg(len);
	thrust::sequence(order, order + len);
	thrust::fill(d.begin(), d.end(), m);
	if (greater)
		thrust::stable_sort_by_key(val, val + len, order,
				thrust::greater<double>());
	else
		thrust::stable_sort_by_key(val, val + len, order);
	thrust::transform(order, order + len, d.begin(), seg.begin(),
			thrust::divides<int>());
	thrust::transform(order, order + len, d.begin(), order,
			thrust::modulus<int>());
	thrust::stable_sort_by_key(seg.begin(), seg.end(), order);
	return order;
}

thrust::device_ptr<int> Dense::sortKeyRow(bool greater) {
	Dense* T = new Dense;
	this->transpose(T);
	thrust::device_ptr<int> order = T->sortKeyCol(greater);
	delete T;
	return order;
}

void Dense::getRow(Dense*d, int r) {
	d->clean();
	d->initial(1, n);
	int thread = THREADS;
	int block = (n + thread - 1) / thread;
	getRowKernel<<<block, thread>>>(cu_val, d->cu_val, r, m, n);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Dense::setRow(Dense*d, int r) {
	int thread = THREADS;
	int block = (n + thread - 1) / thread;
	setRowKernel<<<block, thread>>>(cu_val, d->cu_val, r, m, n);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Dense::diag(Dense*d) {
	d->clean();
	if (m == 1 || n == 1) {
		int l = m * n;
		d->initial(l, l);
		d->setValue(0.0f);
		for (int i = 0; i < l; ++i)
			d->val[i * l + i] = val[i];
	} else {
		d->initial(m, 1);
		for (int i = 0; i < m; ++i)
			d->val[i] = val[i * m + i];
	}
}

void Dense::rbind(Dense*r, Dense*d, double a, double b) {
	if (n != d->n) {
		printf("column not match\n");
		return;
	}
	r->clean();
	r->initial(m + d->m, n);
	dim3 threadsPerBlock(PER_THREADS, PER_THREADS);
	dim3 numBlocks((r->m + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(r->n + threadsPerBlock.y - 1) / threadsPerBlock.y);
	rbindKernel<<<numBlocks, threadsPerBlock>>>(r->cu_val, cu_val, d->cu_val, m,
			d->m, n, a, b);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Dense::rbind(Dense*d, double a, double b) {
	Dense*r = new Dense;
	rbind(r, d, a, b);
	r->clone(this);
	delete r;
}

void Dense::cbind(Dense*r, Dense*d, double a, double b) {
	if (m != d->m) {
		printf("row not match\n");
		return;
	}
	r->clean();
	r->initial(m, n + d->n);
//	checkCudaErrors(cublasDcopy(handle, m*n, cu_val, 1, r->cu_val, 1));
//	checkCudaErrors(
//			cublasDcopy(handle, m*d->n, d->cu_val, 1, r->cu_val + m*n, 1));
	thrust::transform(val, val + length(), r->val, multiply_const(a));
	thrust::transform(d->val, d->val + d->length(), r->val + m * n,
			multiply_const(b));
//	thrust::transform(r->val, r->val + m * n, multiply_const(a));
//	thrust::transform(r->val + m * n, r->val + m * r->n, multiply_const(b));
}

void Dense::cbind(Dense*d, double a, double b) {
	Dense*r = new Dense;
	cbind(r, d, a, b);
	r->clone(this);
	delete r;
}

Dense::~Dense() {
//printf("dense destructor called\n");
	clean();
}

void Sparse::readCSR(const char* filename) {
	FILE* file = fopen(filename, "r");
	char line[100];
	fgets(line, 100, file);
	fscanf(file, "%d %d %d", &m, &n, &nnz);
	initialBoth();
	int i, r, c;
	double v;
	size_t size = sizeof(int) * nnz;
	int*row_host = (int*) malloc(size);
	int*col_host = (int*) malloc(size);
	double*val_host = (double*) malloc(sizeof(double) * nnz);
	if (strstr(line, "pattern") != NULL) {
		for (i = 0; i < nnz; ++i) {
			fscanf(file, "%d %d", &r, &c);
			row_host[i] = r - 1;
			col_host[i] = c - 1;
			val_host[i] = 1;
		}
	} else {
		for (i = 0; i < nnz; ++i) {
			fscanf(file, "%d %d %lf", &r, &c, &v);
			row_host[i] = r - 1;
			col_host[i] = c - 1;
			val_host[i] = v;
		}
	}
	fclose(file);
	uploadCSR(row_host, col_host, val_host);
	delete row_host;
	delete col_host;
	delete val_host;
}

void Sparse::setDiag(Dense*d) {
	nnz = m = n = d->length();
	initialBoth();
	size_t size = sizeof(int) * nnz;
	int*row_host = (int*) malloc(size);
	int*col_host = (int*) malloc(size);
	double*val_host = (double*) malloc(sizeof(double) * nnz);
	for (int i = 0; i < nnz; ++i) {
		row_host[i] = col_host[i] = i;
		val_host[i] = d->val[i];
	}
	uploadCSR(row_host, col_host, val_host);
	delete row_host;
	delete col_host;
	delete val_host;
}

void Sparse::setIdentity(int s) {
	nnz = m = n = s;
	initialBoth();
	size_t size = sizeof(int) * nnz;
	int*row_host = (int*) malloc(size);
	int*col_host = (int*) malloc(size);
	double*val_host = (double*) malloc(sizeof(double) * nnz);
	for (int i = 0; i < nnz; ++i) {
		row_host[i] = col_host[i] = i;
		val_host[i] = 1;
	}
	uploadCSR(row_host, col_host, val_host);
	delete row_host;
	delete col_host;
	delete val_host;
}

void Sparse::readCSC(const char* filename) {
	FILE* file = fopen(filename, "r");
	char line[100];
	fgets(line, 100, file);
	fscanf(file, "%d %d %d", &m, &n, &nnz);
	initialBoth();
	int i, r, c;
	double v;
	size_t size = sizeof(int) * nnz;
	int*row_host = (int*) malloc(size);
	int*col_host = (int*) malloc(size);
	double*val_host = (double*) malloc(sizeof(double) * nnz);
	if (strstr(line, "pattern") != NULL) {
		for (i = 0; i < nnz; ++i) {
			fscanf(file, "%d %d", &r, &c);
			row_host[i] = r - 1;
			col_host[i] = c - 1;
			val_host[i] = 1;
		}
	} else {
		for (i = 0; i < nnz; ++i) {
			fscanf(file, "%d %d %lf", &r, &c, &v);
			row_host[i] = r - 1;
			col[i] = c - 1;
			val[i] = v;
		}
	}
	fclose(file);
	uploadCSC(row_host, col_host, val_host);
	delete row_host;
	delete col_host;
	delete val_host;
}

void Sparse::writeCSR(const char* filename) {
	std::ofstream out(filename);
	out << "%%MatrixMarket matrix coordinate real general" << std::endl;
	out << m << " " << n << " " << nnz << std::endl;
	for (int i = 0; i < m; ++i) {
		int nnz = row[i + 1] - row[i];
		for (int j = 0; j < nnz; ++j)
			out << i + 1 << " " << col[row[i] + j] + 1 << " " << val[row[i] + j]
					<< std::endl;
	}
	out.close();
}

void Sparse::writeCSC(const char* filename) {
	std::ofstream out(filename);
	thrust::device_ptr<int> row(cu_row);
	thrust::device_ptr<int> col(cu_col_index);
	thrust::device_ptr<double> val(trans_val);
	out << "%%MatrixMarket matrix coordinate real general" << std::endl;
	out << m << " " << n << " " << nnz << std::endl;
	for (int i = 0; i < n; ++i) {
		int nnz = col[i + 1] - col[i];
		for (int j = 0; j < nnz; ++j)
			out << row[col[i] + j] + 1 << " " << i + 1 << " " << val[col[i] + j]
					<< std::endl;
	}
	out.close();
}

void Sparse::rowSum(Dense*d) {
	d->clean();
	d->initial(m, 1);
	Dense*one = new Dense;
	one->initial(n, 1);
	one->setValue(1.0f);
	checkCudaErrors(
			cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz,
					&ONE, descr, cu_val, cu_row_index, cu_col, one->cu_val,
					&ZERO, d->cu_val));
	delete one;
}

void Sparse::csrmm2(Dense*r, Dense*d, bool transA, bool transB, double a,
		double b) {
	int m = r->m;
	int n = r->n;
	int k = transA ? this->m : this->n;
	if (!transB) {
		Dense* trans = new Dense;
		d->transpose(trans);
		if (transA) {
			checkCudaErrors(
					cusparseDcsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
							CUSPARSE_OPERATION_TRANSPOSE, m, n, k, nnz, &a,
							descr, trans_val, cu_col_index, cu_row,
							trans->cu_val, n, &b, r->cu_val, m));
		} else {
			checkCudaErrors(
					cusparseDcsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
							CUSPARSE_OPERATION_TRANSPOSE, m, n, k, nnz, &a,
							descr, cu_val, cu_row_index, cu_col, trans->cu_val,
							n, &b, r->cu_val, m));
		}
		delete trans;
	} else {
		if (transA) {
			checkCudaErrors(
					cusparseDcsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
							CUSPARSE_OPERATION_TRANSPOSE, m, n, k, nnz, &a,
							descr, trans_val, cu_col_index, cu_row, d->cu_val,
							n, &b, r->cu_val, m));
		} else {
			checkCudaErrors(
					cusparseDcsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
							CUSPARSE_OPERATION_TRANSPOSE, m, n, k, nnz, &a,
							descr, cu_val, cu_row_index, cu_col, d->cu_val, n,
							&b, r->cu_val, m));
		}
	}
}

void Sparse::csrmv(Dense*y, Dense*x, double alpha, double beta, bool trans) {
	if (trans) {
		checkCudaErrors(
				cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, m, n, nnz,
						&alpha, descr, cu_val, cu_row_index, cu_col, x->cu_val,
						&beta, y->cu_val));
	} else {
		checkCudaErrors(
				cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n,
						nnz, &alpha, descr, cu_val, cu_row_index, cu_col,
						x->cu_val, &beta, y->cu_val));
	}
}

void Sparse::inv(Dense* d) {
	csrilu02Info_t info_LU;
	int size;
	int*buffer;
	int position;
	checkCudaErrors(cusparseCreateCsrilu02Info(&info_LU));
	checkCudaErrors(
			cusparseDcsrilu02_bufferSize(handle, m, nnz, descr, cu_val,
					cu_row_index, cu_col, info_LU, &size));
	checkCudaErrors(cudaMalloc((void** ) &buffer, size));
	checkCudaErrors(
			cusparseDcsrilu02_analysis(handle, m, nnz, descr, cu_val,
					cu_row_index, cu_col, info_LU,
					CUSPARSE_SOLVE_POLICY_NO_LEVEL, &buffer));
	checkCudaErrors(cusparseXcsrilu02_zeroPivot(handle, info_LU, &position));
	checkCudaErrors(
			cusparseDcsrilu02(handle, m, nnz, descr, cu_val, cu_row_index,
					cu_col, info_LU, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer));
	checkCudaErrors(cusparseDestroyCsrilu02Info(info_LU));
	checkCudaErrors(cudaFree(buffer));

	Dense* t = new Dense;
	t->initial(m, m);
	d->clean();
	d->initial(m, m);
	cusparseSolveAnalysisInfo_t info;
	checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info));
	checkCudaErrors(
			cusparseDcsrsm_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m,
					nnz, descr_L, cu_val, cu_row_index, cu_col, info));
	Dense* I = new Dense;
	I->setIdentity(m);
	checkCudaErrors(
			cusparseDcsrsm_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, m,
					&ONE, descr_L, cu_val, cu_row_index, cu_col, info,
					I->cu_val, m, t->cu_val, m));
	checkCudaErrors(
			cusparseDcsrsm_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m,
					nnz, descr_U, cu_val, cu_row_index, cu_col, info));
	checkCudaErrors(
			cusparseDcsrsm_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, m,
					&ONE, descr_U, cu_val, cu_row_index, cu_col, info,
					t->cu_val, m, d->cu_val, m));
	checkCudaErrors(cusparseDestroySolveAnalysisInfo(info));
	delete I;
	delete t;
}

void Sparse::pinv(Dense*r, double tol) {
	Dense* d = new Dense;
	toDense(d);
	d->pinv(r, tol);
	delete d;
}

void Sparse::times(Dense* r, Dense* d, bool transA, bool transB) {
	r->clean();
	int m = transA ? this->n : this->m;
	int n = transB ? d->m : d->n;
	r->initial(m, n);
	csrmm2(r, d, transA, transB, 1.0f, 0.0f);
}

void Sparse::plus(Dense*r, double a, double b, bool trans) {
	Dense*I = new Dense;
	if (trans)
		I->setIdentity(m);
	else
		I->setIdentity(n);
	csrmm2(r, I, trans, false, a, b);
	delete I;
}

void Sparse::plus(Sparse*r, Sparse* s, double a, double b) {
	r->m = m;
	r->n = n;
	r->nnz = 0;
	r->clean();
	checkCudaErrors(
			cudaMalloc((void** ) &r->cu_row_index, sizeof(int) * (r->m + 1)));
	int *nnzTotalDevHostPtr = &s->nnz;
	checkCudaErrors(
			cusparseXcsrgeamNnz(handle, m, n, descr, nnz, cu_row_index, cu_col,
					s->descr, s->nnz, s->cu_row_index, s->cu_col, r->descr,
					r->cu_row_index, nnzTotalDevHostPtr));
	if (NULL != nnzTotalDevHostPtr)
		r->nnz = *nnzTotalDevHostPtr;
	else {
		int base;
		checkCudaErrors(
				cudaMemcpy(&r->nnz, r->cu_row_index + m, sizeof(int),
						cudaMemcpyDeviceToHost));
		checkCudaErrors(
				cudaMemcpy(&base, r->cu_row_index, sizeof(int),
						cudaMemcpyDeviceToHost));
		r->nnz -= base;
	}
	r->initialBoth();
	checkCudaErrors(
			cusparseDcsrgeam(handle, m, n, &a, descr, nnz, cu_val, cu_row_index,
					cu_col, &b, s->descr, s->nnz, s->cu_val, s->cu_row_index,
					s->cu_col, r->descr, r->cu_val, r->cu_row_index,
					r->cu_col));
//cusparseScsrgeam(handle,m,n,&a,descr,nnz,)
}

void Sparse::times(Sparse*nmat, Sparse*mat, bool transA, bool transB) {
	int m, n, k;
	if (!transA) {
		m = this->m;
		k = this->n;
	} else {
		m = this->n;
		k = this->m;
		transpose();
	}
	if (!transB)
		n = mat->n;
	else {
		n = mat->m;
		mat->transpose();
	}
	nmat->m = m;
	nmat->n = n;
	nmat->nnz = 0;
	nmat->clean();
	checkCudaErrors(
			cudaMalloc((void** ) &nmat->cu_row_index,
					sizeof(int) * (nmat->m + 1)));
	int *nnzTotalDevHostPtr = &nmat->nnz;
	if (!transA && !transB)
		checkCudaErrors(
				cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnz,
						cu_row_index, cu_col, descr, mat->nnz,
						mat->cu_row_index, mat->cu_col, descr,
						nmat->cu_row_index, nnzTotalDevHostPtr));
	else if (transA && !transB)
		checkCudaErrors(
				cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnz,
						cu_col_index, cu_row, descr, mat->nnz,
						mat->cu_row_index, mat->cu_col, descr,
						nmat->cu_row_index, nnzTotalDevHostPtr));
	else if (!transA && transB)
		checkCudaErrors(
				cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnz,
						cu_row_index, cu_col, descr, mat->nnz,
						mat->cu_col_index, mat->cu_row, descr,
						nmat->cu_row_index, nnzTotalDevHostPtr));
	else
		checkCudaErrors(
				cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnz,
						cu_col_index, cu_row, descr, mat->nnz,
						mat->cu_col_index, mat->cu_row, descr,
						nmat->cu_row_index, nnzTotalDevHostPtr));

	if (NULL != nnzTotalDevHostPtr)
		nmat->nnz = *nnzTotalDevHostPtr;
	else {
		int base;
		checkCudaErrors(
				cudaMemcpy(&nmat->nnz, nmat->cu_row_index + m, sizeof(int),
						cudaMemcpyDeviceToHost));
		checkCudaErrors(
				cudaMemcpy(&base, nmat->cu_row_index, sizeof(int),
						cudaMemcpyDeviceToHost));
		nmat->nnz -= base;
	}
	nmat->initialBoth();
	if (!transA && !transB)
		checkCudaErrors(
				cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnz,
						cu_val, cu_row_index, cu_col, mat->descr, mat->nnz,
						mat->cu_val, mat->cu_row_index, mat->cu_col,
						nmat->descr, nmat->cu_val, nmat->cu_row_index,
						nmat->cu_col));
	else if (transA && !transB)
		checkCudaErrors(
				cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnz,
						trans_val, cu_col_index, cu_row, mat->descr, mat->nnz,
						mat->cu_val, mat->cu_row_index, mat->cu_col,
						nmat->descr, nmat->cu_val, nmat->cu_row_index,
						nmat->cu_col));
	else if (!transA && transB)
		checkCudaErrors(
				cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnz,
						cu_val, cu_row_index, cu_col, mat->descr, mat->nnz,
						mat->trans_val, mat->cu_col_index, mat->cu_row,
						nmat->descr, nmat->cu_val, nmat->cu_row_index,
						nmat->cu_col));
	else
		checkCudaErrors(
				cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnz,
						trans_val, cu_col_index, cu_row, mat->descr, mat->nnz,
						mat->trans_val, mat->cu_col_index, mat->cu_row,
						nmat->descr, nmat->cu_val, nmat->cu_row_index,
						nmat->cu_col));
}

void Sparse::rowNorm() {
	Dense*d = new Dense;
	rowNorm(d);
	delete d;
}

void Sparse::clean() {
	if (cu_row)
		checkCudaErrors(cudaFree(cu_row));
	if (cu_col)
		checkCudaErrors(cudaFree(cu_col));
	if (cu_val)
		checkCudaErrors(cudaFree(cu_val));
	if (trans_val)
		checkCudaErrors(cudaFree(trans_val));
	if (cu_row_index)
		checkCudaErrors(cudaFree(cu_row_index));
	if (cu_col_index)
		checkCudaErrors(cudaFree(cu_col_index));
	cu_row = 0;
	cu_col = 0;
	cu_val = 0;
	trans_val = 0;
	cu_row_index = 0;
	cu_col_index = 0;
}

void Sparse::initialCSR() {
//printf("m=%d,n=%d,nnz=%d\n", m, n,nnz);
	if (nnz > 0) {
		if (!cu_row)
			checkCudaErrors(cudaMalloc((void** ) &cu_row, sizeof(int) * nnz));
		if (!cu_col)
			checkCudaErrors(cudaMalloc((void** ) &cu_col, sizeof(int) * nnz));
		if (!cu_val)
			checkCudaErrors(
					cudaMalloc((void** ) &cu_val, sizeof(double) * nnz));
	}
	if (m > 0 && !cu_row_index)
		checkCudaErrors(
				cudaMalloc((void** ) &cu_row_index, sizeof(int) * (m + 1)));
}

void Sparse::initialCSC() {
//printf("m=%d,n=%d,nnz=%d\n", m, n, nnz);
	if (nnz > 0) {
		if (!cu_row)
			checkCudaErrors(cudaMalloc((void** ) &cu_row, sizeof(int) * nnz));
		if (!cu_col)
			checkCudaErrors(cudaMalloc((void** ) &cu_col, sizeof(int) * nnz));
		if (!trans_val)
			checkCudaErrors(
					cudaMalloc((void** ) &trans_val, sizeof(double) * nnz));
	}
	if (n > 0 && !cu_col_index)
		checkCudaErrors(
				cudaMalloc((void** ) &cu_col_index, sizeof(int) * (n + 1)));
}

void Sparse::initialBoth() {
//printf("m=%d,n=%d,nnz=%d\n", m, n, nnz);
	if (nnz > 0) {
		if (!cu_row)
			checkCudaErrors(cudaMalloc((void** ) &cu_row, sizeof(int) * nnz));
		if (!cu_col)
			checkCudaErrors(cudaMalloc((void** ) &cu_col, sizeof(int) * nnz));
		if (!cu_val)
			checkCudaErrors(
					cudaMalloc((void** ) &cu_val, sizeof(double) * nnz));
		if (!trans_val)
			checkCudaErrors(
					cudaMalloc((void** ) &trans_val, sizeof(double) * nnz));
	}
	if (m > 0 && !cu_row_index)
		checkCudaErrors(
				cudaMalloc((void** ) &cu_row_index, sizeof(int) * (m + 1)));
	if (n > 0 && !cu_col_index)
		checkCudaErrors(
				cudaMalloc((void** ) &cu_col_index, sizeof(int) * (n + 1)));
	row = thrust::device_pointer_cast(cu_row_index);
	col = thrust::device_pointer_cast(cu_col);
	val = thrust::device_pointer_cast(cu_val);
}

void Sparse::outerTimes(Sparse* s) {
	times(s, this, true, false);
}

void Sparse::outerTimes(Dense*B, Dense*A) {
	Dense*T = new Dense;
	times(T, A, true, false);
	times(B, T, true, true);
	B->transpose();
	delete T;
}

void Sparse::innerTimes(Sparse* s) {
	times(s, this, false, true);
}

void Sparse::innerTimes(Dense*B, Dense*A) {
	Dense*T = new Dense;
	times(T, A, false, false);
	times(B, T, false, true);
	B->transpose();
	delete T;
}

void Sparse::uploadCSR(int* row, int* col, double* val) {
	checkCudaErrors(
			cudaMemcpy(cu_row, row, nnz * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(cu_col, col, nnz * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(cu_val, val, nnz * sizeof(double),
					cudaMemcpyHostToDevice));
	checkCudaErrors(
			cusparseXcoo2csr(handle, cu_row, nnz, m, cu_row_index,
					CUSPARSE_INDEX_BASE_ZERO));
	checkCudaErrors(
			cusparseDcsr2csc(handle, m, n, nnz, cu_val, cu_row_index, cu_col,
					trans_val, cu_row, cu_col_index, CUSPARSE_ACTION_NUMERIC,
					CUSPARSE_INDEX_BASE_ZERO));
}

void Sparse::uploadCSC(int* row, int* col, double* val) {
	checkCudaErrors(
			cudaMemcpy(cu_row, row, nnz * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(cu_col, col, nnz * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(trans_val, val, nnz * sizeof(double),
					cudaMemcpyHostToDevice));
	checkCudaErrors(
			cusparseXcoo2csr(handle, cu_col, nnz, n, cu_col_index,
					CUSPARSE_INDEX_BASE_ZERO));
	checkCudaErrors(
			cusparseDcsr2csc(handle, n, m, nnz, trans_val, cu_col_index, cu_row,
					cu_val, cu_col, cu_row_index, CUSPARSE_ACTION_NUMERIC,
					CUSPARSE_INDEX_BASE_ZERO));
}

void Sparse::diagTimes(Sparse*res, Dense* diag, bool trans) {
	Sparse* diagView = new Sparse;
	diagView->setDiag(diag);
	times(res, diagView, trans, false);
	delete diagView;
}

void Sparse::print() {
	std::cout << "%%MatrixMarket matrix coordinate real general" << std::endl;
	std::cout << m << " " << n << " " << nnz << std::endl;
	for (int i = 0; i < m; ++i) {
		int nnz = row[i + 1] - row[i];
		for (int j = 0; j < nnz; ++j)
			std::cout << i + 1 << " " << col[row[i] + j] + 1 << " "
					<< val[row[i] + j] << std::endl;
	}
}

void Sparse::printFull() {
	Dense*d = new Dense;
	toDense(d);
	d->print();
	delete d;
}

void Sparse::transpose() {
	if (!cu_col_index) {
		checkCudaErrors(
				cudaMalloc((void** ) &cu_col_index, sizeof(int) * (n + 1)));
		checkCudaErrors(cudaMalloc((void** ) &trans_val, sizeof(double) * nnz));
		checkCudaErrors(
				cusparseDcsr2csc(handle, m, n, nnz, cu_val, cu_row_index,
						cu_col, trans_val, cu_row, cu_col_index,
						CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));
	}
}

void Sparse::selfTimes(Dense*r, Dense* d) {
	times(r, d, false, true);
	eTimes(r);
	rowSum(r);
}

void Sparse::toDense(Dense* dense) {
	dense->clean();
	dense->initial(m, n);
	checkCudaErrors(
			cusparseDcsr2dense(handle, m, n, descr, cu_val, cu_row_index,
					cu_col, dense->cu_val, dense->m));
}

double Sparse::getElem(int i, int j) {
	int nnz = row[i + 1] - row[i];
	int k;
	for (k = 0; k < nnz && col[row[i] + k] < j; ++k)
		;
	if (k < nnz && col[row[i] + k] == j)
		return val[row[i] + k];
	return 0.0f;
}

Sparse::~Sparse() {
//printf("sparse destructor called\n");
	clean();
}

void Sparse::rowVec(Dense*d, int r, bool column) {
	d->clean();
	if (column)
		d->initial(n, 1);
	else
		d->initial(1, n);
	d->setValue(0.0f);
	int thread = THREADS;
	int block = (n + thread - 1) / thread;
	subvecKernel<<<block, thread>>>(cu_row_index, cu_col, r, cu_val, d->cu_val);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Sparse::rowVec(Dense*d, int r) {
	rowVec(d, r, false);
}

void Sparse::colVec(Dense*d, int c, bool row) {
	d->clean();
	if (row)
		d->initial(1, m);
	else
		d->initial(m, 1);
	d->setValue(0.0f);
	int thread = THREADS;
	int block = (m + thread - 1) / thread;
	subvecKernel<<<block, thread>>>(cu_col_index, cu_row, c, trans_val,
			d->cu_val);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Sparse::colVec(Dense*d, int c) {
	colVec(d, c, false);
}

void Sparse::rowNorm(Dense*r) {
	r->clean();
	r->initial(m, 1);
	int thread = THREADS;
	int block = (m + thread - 1) / thread;
	rowNormSPKernel<<<block, thread>>>(cu_row_index, cu_val, r->cu_val, m);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(
			cusparseDcsr2csc(handle, m, n, nnz, cu_val, cu_row_index, cu_col,
					trans_val, cu_row, cu_col_index, CUSPARSE_ACTION_NUMERIC,
					CUSPARSE_INDEX_BASE_ZERO));
}

void Sparse::rowMultiply(double alpha) {
	int thread = THREADS;
	int block = (m + thread - 1) / thread;
	rowMulKernel<<<block, thread>>>(cu_row_index, cu_val, alpha, m);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Sparse::colSum(Dense*d) {
	transpose();
	d->clean();
	d->initial(1, n);
	int thread = THREADS;
	int block = (m + thread - 1) / thread;
	rowSumKernel<<<block, thread>>>(cu_col_index, trans_val, n, d->cu_val);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Sparse::eTimes(Dense* d) {
	dim3 threadsPerBlock(PER_THREADS, PER_THREADS);
	dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(n + threadsPerBlock.y - 1) / threadsPerBlock.y);
	eTimesSPKernel<<<numBlocks, threadsPerBlock>>>(cu_row_index, cu_col, cu_val,
			d->cu_val, m, n);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Sparse::eTimes(Dense*r, Sparse*s) {
	int block = (m + THREADS - 1) / THREADS;
	r->clean();
	r->initial(s->n, s->m);
	etimesKernel<<<block, THREADS>>>(cu_row_index, cu_col, cu_val,
			s->cu_row_index, s->cu_col, s->cu_val, r->cu_val, m, n);
	checkCudaErrors(cudaDeviceSynchronize());
	r->transpose();
}

void Dense::signPlus(Dense*s, double v) {
	int thread = THREADS;
	int len = length();
	int block = (len + thread - 1) / thread;
	signPlusKernel<<<block, thread>>>(cu_val, s->cu_val, v, len);
	checkCudaErrors(cudaDeviceSynchronize());
}
//struct square_root: public thrust::unary_function<float, float> {
//	__host__ __device__
//	float operator()(float x) const {
//		return sqrtf(x);
//	}
//};
void Dense::square_root() {
	pow(0.5f);
}
struct power: public thrust::unary_function<double, double> {
	const double a;
	power(double _a) :
			a(_a) {
	}
	__host__ __device__
	double operator()(const double &x) const {
		return powf(x, a);
	}
};
void Dense::pow(double ind) {
	thrust::transform(val, val + length(), val, power(ind));
}

void Dense::project() {
	int thread = THREADS;
	int len = length();
	int block = (len + thread - 1) / thread;
	projectKernel<<<block, thread>>>(cu_val, len);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Dense::shrink(double tau) {
	dim3 threadsPerBlock(PER_THREADS, PER_THREADS);
	dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(n + threadsPerBlock.y - 1) / threadsPerBlock.y);
	shrinkKernel<<<numBlocks, threadsPerBlock>>>(tau, cu_val, m, n);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Dense::repmat(Dense*d, int m, int n) {
	d->clean();
	d->initial(m * this->m, n * this->n);
	dim3 threadsPerBlock(PER_THREADS, PER_THREADS);
	dim3 numBlocks((d->m + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(d->n + threadsPerBlock.y - 1) / threadsPerBlock.y);
	repmatKernel<<<numBlocks, threadsPerBlock>>>(d->cu_val, cu_val, d->m, d->n,
			this->m, this->n);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Dense::sub(Dense*d, int rs, int re, int cs, int ce) {
	d->clean();
	int m = re - rs;
	int n = ce - cs;
	d->initial(m, n);
	dim3 threadsPerBlock(PER_THREADS, PER_THREADS);
	dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(n + threadsPerBlock.y - 1) / threadsPerBlock.y);
	subKernel<<<numBlocks, threadsPerBlock>>>(cu_val, d->cu_val, rs, re, cs, ce,
			this->m);
	checkCudaErrors(cudaDeviceSynchronize());
}

