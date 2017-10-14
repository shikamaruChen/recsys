/*
 * prism.cpp
 *
 *  Created on: 2017年8月25日
 *      Author: yifan
 */

#include "prism.h"

void Prism::laplace() {
	Dense*row = new Dense;

	S->plus(Sy, S, 0.5, 0.5, true, false);
	Sy->rowSum(row);
	row->diag(D);
	D->plus(L, S, 1.0, -1.0, false, false);
//
//	S->colSum(col);
//	row->plus(v, col, 0.5f, 0.5f, true, false);
//	v->diag(D);
//	D->plus(L, S, 1.0f, -0.5f, false, false);
//	L->plus(S, 1.0f, -0.5f, true);	//L -= 0.5 S

	delete row;
}

void Prism::calcQ() {
	Dense*n = new Dense;
	Dense*t = new Dense;
	FV->rowSquare(n);
	n->repmat(t, 1, Ni);
	t->plus(Q, t, 1.0f, 1.0f, true, false);
	Q->times(FV, FV, 1.0f, -2.0f, false, true);
	delete n;
	delete t;
}

double Prism1::object() {
	Dense* term = new Dense;
	R->times(term, S, false, false);
	R->plus(term, 1.0, -1.0, false);
	double obj = term->frobenius() / 2;
	FV->rtimes(term, V, 1.0, false, true);
	obj += alpha / 4 * term->frobenius();
	FV->rtimes(term, L, 1.0, true, false);
	obj += beta / 4 * term->trace(FV);
	obj += lambda1 * S->norm1();
	obj += lambda2 / 2 * V->frobenius();
	return obj;
}

void Prism1::updateS() {
	Dense*numerator = new Dense;
	Dense*denominator = new Dense;
	RR->rtimes(numerator, S, 1.0, false, false);
	RR->plus(denominator, Q, 1.0, beta / 4, false, false);
	denominator->plus(1.0, lambda1);
	S->eTimes(numerator);
	S->eDiv(denominator);
	delete numerator;
	delete denominator;
}

void Prism1::updateV() {
	//numerator
	Dense*numerator = new Dense;
	Dense*denominator = new Dense;
	Dense*term = new Dense;
	F->times(FV, V, false, false);
	F->times(denominator, FV, true, false); // denominator=FFV
	V->rtimes(term, V, 1.0, true, false);
	denominator->rtimes(numerator, term, 1.0, false, false);
//	numerator->rtimes(term, 1.0, false, false);
	F->times(term, Sy, true, false);
	numerator->times(term, FV, alpha, beta, false, false);
	//denominator
	F->times(term, D, true, false);
	denominator->times(FD, FV, alpha, beta, false, false);
	denominator->plus(V, 1.0, lambda2, false);
	//update
	V->eTimes(numerator);
	V->eDiv(denominator);
	F->times(FV, V, false, false);
	delete numerator;
	delete denominator;
	delete term;
}

void Prism::learn() {
	Sparse*sp = new Sparse;
	R->outerTimes(sp);
	sp->toDense(RR);
	delete sp;

	S->initial(Ni, Ni);
	S->setRandom();
	S->setDiagValue(0.0f);
	laplace();
	V->initial(Nf, k);
	V->setRandom();

	for (int iter = 1; iter <= maxiter; ++iter) {
		printf("---------- iter:%d ------------\n", iter);
		updateV();
		calcQ();
		updateS();
		laplace();
		printf("obj=%f\n", object());
	}
	R->times(pR, S, false, false);
}

