/*
 * lce.cpp
 *
 *  Created on: 2017年2月20日
 *      Author: yifan
 */

#include "lce.h"

__global__ void selectKernel(float*S, int*index, int m, int n, int k);

void LCE::laplace(int num) {
	Sparse*s = new Sparse;
	F->innerTimes(s);
	s->toDense(S);
//	S->print();
	delete s;
	thrust::device_ptr<int> order = S->sortKeyRow(true);
	S->setValue(0.0f);
	dim3 threadsPerBlock(PER_THREADS, PER_THREADS);
	dim3 numBlocks((S->m + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(num + threadsPerBlock.y - 1) / threadsPerBlock.y);
	selectKernel<<<numBlocks, threadsPerBlock>>>(S->cu_val, order.get(), S->m,
			S->n, num);
	checkCudaErrors(cudaDeviceSynchronize());
	thrust::device_free(order);

	Dense*row = new Dense;
	Dense*col = new Dense;
	Dense*v = new Dense;

	S->rowSum(row);
	S->colSum(col);
	row->plus(v, col, 0.5f, 0.5f, true, false);
	v->diag(D);
	D->plus(L, S, 1.0f, -0.5f, false, false);
	L->plus(S, 1.0f, -0.5f, true);	//L -= 0.5 S
	D->plus(S, L, 1.0f, -1.0f, false, false);
//	L->print();
	delete row;
	delete col;
	delete v;
//	S->print();
}

void LCE::learn() {
	U->initial(Nu, k);
	V->initial(Ni, k);
	W->initial(Nf, k);
	U->setRandom();
	V->setRandom();
	W->setRandom();
	laplace(2);
	for (int iter = 1; iter <= maxiter; ++iter) {
		printf("---------- iter:%d ------------\n", iter);
		updateV();
		V->rtimes(VV, V, 1.0f, true, false);
		updateW();
		updateU();
		printf("obj=%f\n", object());
	}
	predict();
}

double LCE::object() {
	Dense*t = new Dense;
	U->rtimes(t, V, 1.0f, false, true);
	R->plus(t, 1.0f, -1.0f, false);
	double obj = t->square() * alpha / 2;
	V->rtimes(t, W, 1.0f, false, true);
	F->plus(t, 1.0f, -1.0f, false);
	obj += t->square() * (1 - alpha) / 2;
	V->rtimes(t, L, 1.0f, true, false);
	obj += beta * t->trace(V) / 2;
	obj += lambda / 2 * (U->square() + V->square() + W->square());
	return obj;
}

void LCE::predict() {
	Dense*FW = new Dense;
	Dense*WW = new Dense;
	Dense*d = new Dense;
	F->times(FW, W, false, false);
	W->rtimes(WW, W, 1.0f, true, false);
	for (int i = 0; i < 10; ++i) {
		V->rtimes(d, WW, 1.0f, false, false);
		V->eTimes(FW);
		V->eDiv(d);
	}
	U->rtimes(pR, V, 1.0f, false, true);
	delete FW;
	delete WW;
	delete d;
}

void LCE::record(const char*filename) {
	printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold, k,
			alpha, beta, lambda, HR[0] / test->nnz, ARHR[0] / test->nnz,
			HR[1] / test->nnz, ARHR[1] / test->nnz, HR[2] / test->nnz,
			ARHR[2] / test->nnz, HR[3] / test->nnz, ARHR[3] / test->nnz);
	FILE*file = fopen(filename, "a");
	fprintf(file, "%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold,
			k, alpha, beta, lambda, HR[0] / test->nnz, ARHR[0] / test->nnz,
			HR[1] / test->nnz, ARHR[1] / test->nnz, HR[2] / test->nnz,
			ARHR[2] / test->nnz, HR[3] / test->nnz, ARHR[3] / test->nnz);
	fclose(file);
}

void LCE::updateU() {
	Dense*n = new Dense;
	Dense*d = new Dense;

	R->times(n, V, false, false);
	U->rtimes(d, VV, 1.0f, false, false);
	d->plus(U, alpha, lambda, false);

	U->eTimes(n);
	U->times(alpha);
	U->eDiv(d);

	delete n;
	delete d;
}

void LCE::updateV() {
	Dense*n = new Dense;
	Dense*d = new Dense;
	Dense*t = new Dense;
	float gamma = 1 - alpha;

	R->times(n, U, true, false);
	F->csrmm2(n, W, false, false, gamma, alpha);
//	F->times(t, W, false, false);
//	n->plus(t, alpha, gamma, false);
	n->times(S, V, 1.0f, beta, false, false);
//	n->times(S, V, 1.0f, beta / 2, true, false);

	V->rtimes(d, U, 1.0f, false, true);
	d->rtimes(U, 1.0f, false, false);
	V->rtimes(t, W, 1.0f, false, true);
	d->times(t, W, alpha, gamma, false, false);
	d->times(D, V, 1.0f, beta, false, false);
	d->plus(V, 1.0f, lambda, false);

	V->eTimes(n);
	V->eDiv(d);

	delete n;
	delete d;
	delete t;
}
void LCE::updateW() {
	Dense*n = new Dense;
	Dense*d = new Dense;
	float gamma = 1 - alpha;

	F->times(n, V, true, false);
	W->rtimes(d, VV, 1.0f, false, false);
	d->plus(W, gamma, lambda, false);

	W->eTimes(n);
	W->times(gamma);
	W->eDiv(d);

	delete n;
	delete d;
}

//S = new Dense;
//U = new Dense;
//V = new Dense;
//W = new Dense;
//WW = new Dense;
LCE::~LCE() {
	if (D)
		delete D;
	if (S)
		delete S;
	if (L)
		delete L;
	if (U)
		delete U;
	if (V)
		delete V;
	if (W)
		delete W;
	if (VV)
		delete VV;
}
