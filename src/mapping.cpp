/*
 * mapping.cpp
 *
 *  Created on: 2017年2月17日
 *      Author: yifan
 */
#include "mapping.h"

Mapping::~Mapping() {
	if (FW)
		delete FW;
	if (RR)
		delete RR;
	if (S)
		delete S;
	if (L)
		delete L;
	if (D)
		delete D;
	if (W)
		delete W;
}

void Mapping::record(const char* filename) {
	printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold, k,
			alpha, beta, lambda,  HR[0] / test->nnz, ARHR[0] / test->nnz,
			HR[1] / test->nnz, ARHR[1] / test->nnz, HR[2] / test->nnz,
			ARHR[2] / test->nnz, HR[3] / test->nnz, ARHR[3] / test->nnz);
	FILE*file = fopen(filename, "a");
	fprintf(file, "%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
			fold, k, alpha, beta, lambda,  HR[0] / test->nnz,
			ARHR[0] / test->nnz, HR[1] / test->nnz, ARHR[1] / test->nnz,
			HR[2] / test->nnz, ARHR[2] / test->nnz, HR[3] / test->nnz,
			ARHR[3] / test->nnz);
	fclose(file);
}

void Mapping::learn() {
	Sparse* r = new Sparse;
	R->outerTimes(r);
	r->toDense(RR);
	delete r;
	S->initial(Ni, Ni);
	S->setRandom();
	S->setDiagValue(0.0f);
	F->toDense(FW);
	printf("---------- iter:1 ------------\n");
	updateS();
//	printf("obj=%f\n", object());
	for (int iter = 2; iter <= maxiter; ++iter) {
		printf("---------- iter:%d ------------\n", iter);
		updateW();
		updateS();
	}
	R->times(pR, S, false, false);
}

double Mapping::object() {
	Dense* t = new Dense;
	R->times(t, S, false, false);
	R->plus(t, 1.0f, -1.0f, false);
	double obj = t->square() / 2;
	FW->rtimes(t, S, 1.0f, true, false);
	t->plus(FW, -1.0f, 1.0f, true);
	obj += alpha * t->square() / 2;
	obj += beta * S->norm1();
	obj += lambda / 2 * W->square();
	delete t;
	return obj;
}
void Mapping::updateW() {
	Dense*d1 = new Dense;
	Dense*d2 = new Dense;
	F->times(d1, L, true, false);
	F->times(d2, d1, true, true);
	d2->eig();
	d2->sub(W, 0, Nf, 0, k);
//	W->print();
	F->times(FW, W, false, false);
	delete d1;
	delete d2;
}

void Mapping::updateS() {
	Dense*n = new Dense;
	Dense*d = new Dense;

	FW->rtimes(n, FW, 1.0f, false, true);
	n->plus(RR, alpha, 1.0f, false);
	for (int i = 0; i < 10; ++i) {
		n->rtimes(d, S, 1.0f, false, false);
		d->plus(1.0f, beta);
		S->eTimes(n);
		S->eDiv(d);
		printf("obj=%f\n", object());
	}
	laplace();
	delete n;
	delete d;
}

void Mapping::laplace() {

	Dense*row = new Dense;
	Dense*col = new Dense;
	Dense*v = new Dense;
	//	S->plus(sS, S, 0.5f, 0.5f, true, false);
	//	S->rowSum(v);
	S->rowSum(row);
	S->colSum(col);
	row->plus(v, col, 0.5f, 0.5f, true, false);
	v->diag(D);
	D->plus(L, S, 1.0f, -0.5f, false, false);
	L->plus(S, 1.0f, -0.5f, true);	//L -= 0.5 S
	//L->plus(S, -0.5f);
	delete row;
	delete col;
	delete v;

}

