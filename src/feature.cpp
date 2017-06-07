/*
 * feature.cpp
 *
 *  Created on: 2017年2月16日
 *      Author: yifan
 */
#include "feature.h"

double Feature::object() {
	Dense* term = new Dense;
	R->times(term, S, false, false);
	R->plus(term, 1.0f, -1.0f, false);
	double obj = term->square() / 2;
	F->times(term, S, true, false);
	F->plus(term, 1.0f, -1.0f, true);
	obj += term->square() * alpha / 2;
	FW->rtimes(term, L, 1.0f, true, false);
	obj += mu * term->trace(FW);
	obj += beta * S->norm1();
	obj += lambda / 2 * S->square();
	delete term;
	return obj;
}

void Feature::laplace() {
	Dense*row = new Dense;
	Dense*col = new Dense;
	Dense*v = new Dense;
	Dense*D = new Dense;

	S->rowSum(row);
	S->colSum(col);
	row->plus(v, col, 0.5f, 0.5f, true, false);
	v->diag(D);
	D->plus(L, S, 1.0f, -0.5f, false, false);
	L->plus(S, 1.0f, -0.5f, true);	//L -= 0.5 S

	delete D;
	delete row;
	delete col;
	delete v;
}

void Feature::updateS() {
	Dense* t = new Dense;
	Dense* n = new Dense;
	Dense* d = new Dense;
	Dense*Q = new Dense;

	d->initial(Ni, Ni);
	printf("update S\n");
	for (int iter = 0; iter < 20; ++iter) {
		printf("------ iter:%d ------\n", iter);
		d->setValue(1.0f);
		d->plus(S, lambda, beta, false);
		d->times(RF, S, 1.0f, 1.0f, false, false);
		FW->rowSquare(n);
		n->repmat(t, 1, Ni);
		t->plus(Q, t, 1.0f, 1.0f, true, false);
		Q->times(FW, FW, 1.0f, -2.0f, false, true);
		d->plus(Q, 1.0f, mu / 2, false);
		S->eTimes(RF);
		S->eDiv(d);
	}
	laplace();
	printf("obj=%f\n", object());
	delete t;
	delete d;
	delete n;
	delete Q;
}

void Feature::updateW() {
	Dense*d1 = new Dense;
	Dense*d2 = new Dense;
	F->times(d1, L, true, false);
	F->times(d2, d1, true, true);
	d2->eig();
	d2->sub(W, 0, Nf, 0, k);
	F->times(FW, W, false, false);
	delete d1;
	delete d2;
}

void Feature::learn() {
	Sparse* r = new Sparse;
	Sparse*f = new Sparse;
	Dense*RR = new Dense;
	Dense*FF = new Dense;
	R->outerTimes(r);
	r->toDense(RR);
	F->innerTimes(f);
	f->toDense(FF);
	RR->plus(RF, FF, 1.0f, alpha, false, false);
	delete r;
	delete f;
	delete RR;
	delete FF;

	S->initial(Ni, Ni);
	S->setRandom();
	S->setDiagValue(0.0f);

	F->toDense(FW);
	printf("---------- iter:1 ------------\n");
	updateS();
	for (int iter = 2; iter <= maxiter; ++iter) {
		printf("---------- iter:%d ------------\n", iter);
		updateW();
		updateS();
	}
	R->times(pR, S, false, false);
}

void Feature::record(const char*filename) {
	printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold, k,
			alpha, beta, lambda, mu, HR[0] / test->nnz, ARHR[0] / test->nnz,
			HR[1] / test->nnz, ARHR[1] / test->nnz, HR[2] / test->nnz,
			ARHR[2] / test->nnz, HR[3] / test->nnz, ARHR[3] / test->nnz);
	FILE*file = fopen(filename, "a");
	fprintf(file, "%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
			fold, k, alpha, beta, lambda, mu, HR[0] / test->nnz,
			ARHR[0] / test->nnz, HR[1] / test->nnz, ARHR[1] / test->nnz,
			HR[2] / test->nnz, ARHR[2] / test->nnz, HR[3] / test->nnz,
			ARHR[3] / test->nnz);
	fclose(file);
}

void Feature::print() {
	printf("k=%d\n", k);
	printf("alpha=%f\n", alpha);
	printf("beta=%f\n", beta);
	printf("lambda=%f\n", lambda);
	printf("mu=%f\n", mu);
}

Feature::~Feature() {
	if (FW)
		delete FW;
	if (S)
		delete S;
	if (L)
		delete L;
	if (W)
		delete W;
	if (RF)
		delete RF;
}
