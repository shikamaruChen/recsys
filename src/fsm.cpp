/*
 * FSMrec.cpp
 *
 *  Created on: 2017年5月17日
 *      Author: yifan
 */

#include "fsm.h"

double FSMrec::object() {
	double obj = tR->frobenius() / 2;
	obj += lambda / 2 * w->norm2();
	return obj;
}

void FSMrec::laplace(Dense* L) {
	Dense*Q = new Dense;
	Dense* temp = new Dense;
	R->times(Q, tR, true, false);
	Q->plus(1.0f, alpha);
	Q->clone(L);
	//tildeS->plus(Sp, 1.0f);
	L->eTimes(S);
	L->rowSum(temp);
	temp->diag(L);
	L->plus(Q, -1.0f, 1.0f, false);
	delete temp;
	delete Q;
}

void FSMrec::partial() {
	Dense*L = new Dense;
	Dense*r = new Dense;
	laplace(L);
	nF->selfTimes(dw, L);
	dw->plus(w, 1.0f, lambda, false);
	delete L;
	delete r;
}

void FSMrec::tildeCalc() {
	R->times(tR, S, false, true);
	R->plus(tR, -1.0f, 1.0f, false);
}

void FSMrec::prosimCalc(Dense* w) {
	Dense* t = new Dense;
	w->clone(t);
	t->square_root();
	F->diagTimes(nF, t, false);
	nF->rowNorm(t);
	Sparse*temp = new Sparse;
	nF->innerTimes(temp);
	temp->toDense(S);
	S->setDiagValue(0.0f);
	F->diagTimes(nF, t, true);
	delete t;
	delete temp;
}

void FSMrec::learn() {
	int iter;
	float a;
	float b = 0.1f;
	float obj0, obj1;
	float n2;
	float tol = 0.00001;
	int i;
	Dense* w0 = new Dense;
	Dense* w1 = new Dense;
	w->initial(Nf, 1);
	w->setRandom();
	w->clone(w0);
	for (iter = 1; iter <= maxiter; ++iter) {
		prosimCalc(w);
		tildeCalc();
		obj0 = object();
		printf("---------- iter:%d ------------\n", iter);
		printf("obj=%f\n", obj0);
		//tildeR->plus(Rb, train, &ONE, &NEG_ONE);
		partial();
		a = 100.0f / dw->norm2();
		if (a > 1)
			a = 1;
		a = -a;
		w0->plus(dw, 1.0f, -1.0f, false);
		//w0->plus(dw, -1.0f);
		w0->project();
		prosimCalc(w0);
		tildeCalc();
		obj1 = object();
		w0->plus(w1, w, 1.0f, -1.0f, false, false);
		//w0->plus(w2, w, 1.0f, -1.0f);
		n2 = w1->dot(dw);
		if (obj1 - obj0 > 0.01 * n2) {
			for (i = 0; i < 20; ++i) {
				a *= b;
				w->plus(w0, dw, 1.0f, a, false, false);
//				w->plus(w0, dw, 1.0f, a);
				w0->project();
				prosimCalc(w0);
				tildeCalc();
				obj1 = object();
				w0->plus(w1, w, 1.0f, -1.0f, false, false);
				n2 = w1->dot(dw);
				if (obj1 - obj0 <= 0.01 * n2)
					break;
			}
		}
		w0->plus(w1, w, 1.0f, -1.0f, false, false);
		n2 = w1->norm2();
		printf("norm dw=%f\n", n2);
		if (n2 < tol)
			break;
		//if (obj1>=obj0) break;
		w0->copyto(w);
		//printf("norm w=%f\n", w->norm2());
	}
	R->times(pR, S, false, false);
	delete w0;
	delete w1;
}

void FSMrec::record(const char*filename) {
	printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold, alpha, lambda,
			HR[0] / test->nnz, ARHR[0] / test->nnz, HR[1] / test->nnz,
			ARHR[1] / test->nnz, HR[2] / test->nnz, ARHR[2] / test->nnz,
			HR[3] / test->nnz, ARHR[3] / test->nnz);
	FILE*file = fopen(filename, "a");
	fprintf(file, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold, alpha,
			lambda, HR[0] / test->nnz, ARHR[0] / test->nnz, HR[1] / test->nnz,
			ARHR[1] / test->nnz, HR[2] / test->nnz, ARHR[2] / test->nnz,
			HR[3] / test->nnz, ARHR[3] / test->nnz);
	fclose(file);
}

void FSMrec::model(const char*filename) {
	FILE*file = fopen(filename, "a");
	int len = w->length();
	for (int i = 0; i < len; i++) {
		double v = w->val[i];
		fprintf(file, "%f ", v);
	}
	fprintf(file, "\n");
	fclose(file);
}
//nF = new Sparse();
//tR = new Dense();
//tS = new Dense();
//w = new Dense();
//dw = new Dense();
FSMrec::~FSMrec() {
	// TODO Auto-generated destructor stub
	if (nF)
		delete nF;
	if (tR)
		delete tR;
	if (S)
		delete S;
	if (w)
		delete w;
	if (dw)
		delete dw;
}

