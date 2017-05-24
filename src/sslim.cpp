/*
 * sslim.cpp
 *
 *  Created on: 2017年2月16日
 *      Author: yifan
 */

#include "sslim.h"
#include <set>

#include <algorithm>
#define THREADS 512
#define PER_THREADS 16

void SSLIM::record(const char* filename) {
	FILE*file = fopen(filename, "a");
	if (LOO) {
		printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold, alpha,
				beta, lambda, HR[0] / test->nnz, ARHR[0] / test->nnz,
				HR[1] / test->nnz, ARHR[1] / test->nnz, HR[2] / test->nnz,
				ARHR[2] / test->nnz, HR[3] / test->nnz, ARHR[3] / test->nnz);
		fprintf(file, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold,
				alpha, beta, lambda, HR[0] / test->nnz, ARHR[0] / test->nnz,
				HR[1] / test->nnz, ARHR[1] / test->nnz, HR[2] / test->nnz,
				ARHR[2] / test->nnz, HR[3] / test->nnz, ARHR[3] / test->nnz);
	} else {
		printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold, alpha,
				beta, lambda, REC[0] / test->nnz, REC[1] / test->nnz,
				REC[2] / test->nnz, REC[3] / test->nnz, DCG[0] / test->nnz,
				DCG[1] / test->nnz, DCG[2] / test->nnz, DCG[3] / test->nnz);
		fprintf(file, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold,
				alpha, beta, lambda, REC[0] / test->nnz, REC[1] / test->nnz,
				REC[2] / test->nnz, REC[3] / test->nnz, DCG[0] / test->nnz,
				DCG[1] / test->nnz, DCG[2] / test->nnz, DCG[3] / test->nnz);
	}
	fclose(file);
}

SSLIM::~SSLIM() {
	if (S)
		delete S;
	if (Q)
		delete Q;
}

double SSLIM1::object() {
	Dense* term = new Dense;
	R->times(term, S, false, false);
	R->plus(term, 1.0f, -1.0f, false);
	float nf = term->norm2();
	double obj = nf * nf / 2;
	F->times(term, S, true, false);
	F->plus(term, 1.0f, -1.0f, true);
	nf = term->norm2();
	obj += alpha * nf * nf / 2;
	obj += beta * S->norm1();
	nf = S->norm2();
	obj += lambda / 2 * nf * nf;
	delete term;
	return obj;
}

void SSLIM1::learn() {
	Sparse* RR = new Sparse;
	Sparse*FF = new Sparse;
	Dense*dRR = new Dense;
	Dense*dFF = new Dense;
	Dense*RF = new Dense;
	Dense*d = new Dense;
	d->initial(Ni, Ni);
	R->outerTimes(RR);
	RR->toDense(dRR);
	F->innerTimes(FF);
	FF->toDense(dFF);
	dRR->plus(RF, dFF, 1.0f, alpha, false, false);
	delete RR;
	delete FF;
	delete dRR;
	delete dFF;

	S->initial(Ni, Ni);
	S->setRandom();
	S->setDiagValue(0.0f);
	printf("obj=%f\n", object());

	for (int iter = 1; iter <= maxiter; ++iter) {
		printf("---------- iter:%d ------------\n", iter);
		d->setValue(1.0f);
		d->plus(S, lambda, beta, false);
		d->times(RF, S, 1.0f, 1.0f, false, false);
		S->eTimes(RF);
		S->eDiv(d);
		printf("obj=%f\n", object());
	}
	delete d;
	delete RF;
//	S->print();
	R->times(pR, S, false, false);
}

void SSLIM2::learn() {
	Sparse* RR = new Sparse;
	Sparse*FF = new Sparse;
	Dense*dRR = new Dense;
	Dense*dFF = new Dense;
	Dense*d = new Dense;
	Dense*n = new Dense;
	d->initial(Ni, Ni);
	n->initial(Ni, Ni);
	R->outerTimes(RR);
	RR->toDense(dRR);
	F->innerTimes(FF);
	FF->toDense(dFF);
	S->initial(Ni, Ni);
	S->setRandom();
	S->setDiagValue(0.0f);
	Q->initial(Ni, Ni);
	Q->setRandom();
	Q->setDiagValue(0.0f);
	delete RR;
	delete FF;
	printf("obj=%f\n", object());
	for (int iter = 1; iter <= maxiter; ++iter) {
		printf("---------- iter:%d ------------\n", iter);
		//update S
		dRR->copyto(n);
		n->plus(Q, 1.0f, beta, false);
		d->setValue(1.0f);
		d->plus(S, lambda, beta, false);
		d->times(dRR, S, 1.0f, 1.0f, false, false);
		S->eTimes(n);
		S->eDiv(d);
		// update Q
		dFF->copyto(n);
		n->plus(S, alpha, beta, false);
		d->setValue(1.0f);
		d->plus(Q, lambda, beta, false);
		d->times(dFF, Q, 1.0f, alpha, false, false);
		printf("obj=%f\n", object());
	}
	R->times(pR, S, false, false);
	delete dRR;
	delete dFF;
	delete d;
	delete n;
}

double SSLIM2::object() {
	Dense* term = new Dense;
	R->times(term, S, false, false);
	R->plus(term, 1.0f, -1.0f, false);
	float nf = term->norm2();
	double obj = nf * nf / 2;
	F->times(term, Q, true, false);
	F->plus(term, 1.0f, -1.0f, true);
	nf = term->norm2();
	obj += alpha * nf * nf / 2;
	S->plus(term, Q, 1.0f, -1.0f, false, false);
	nf = term->norm2();
	obj += beta * nf * nf / 2;
	obj += lambda * S->norm1();
	obj += lambda * Q->norm1();
	delete term;
	return obj;
}

